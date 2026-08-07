"""BM25 sparse index with a tokenizer shared by build time and query time.

Two backends are supported:

``bm25s``
    The current backend. Scoring runs on a scipy sparse matrix and only the
    top-n documents are materialised, so query cost does not grow with the
    number of chunks the way a pure-Python scan does.

``rank_bm25``
    Legacy backend, kept so indexes published before the tokenizer fix still
    load. It scores every document on every query; over the 338k-chunk corpus
    that costs roughly 1.9s per query.

Both are reached through :class:`SparseIndex`, so the retriever does not care
which one is loaded.
"""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path
from typing import Any

from egov_bot.utils.normalizer import BM25_TOKENIZER_VERSION, tokenize_bm25

logger = logging.getLogger(__name__)

BACKEND_BM25S = "bm25s"
BACKEND_LEGACY = "rank_bm25"


class SparseIndex:
    """A BM25 index that tokenizes queries exactly as it tokenized documents."""

    def __init__(
        self,
        backend: Any,
        kind: str,
        doc_count: int,
        tokenizer_version: str = BM25_TOKENIZER_VERSION,
    ) -> None:
        self.backend = backend
        self.kind = kind
        self.doc_count = doc_count
        self.tokenizer_version = tokenizer_version

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, texts: list[str]) -> SparseIndex:
        """Build a fresh index from raw chunk texts."""
        import bm25s

        corpus_tokens = [tokenize_bm25(text) for text in texts]
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        return cls(retriever, BACKEND_BM25S, doc_count=len(corpus_tokens))

    @classmethod
    def from_legacy(cls, bm25: Any) -> SparseIndex:
        """Wrap a pickled ``rank_bm25`` object built by an older revision."""
        doc_count = len(getattr(bm25, "doc_freqs", []) or [])
        return cls(bm25, BACKEND_LEGACY, doc_count=doc_count, tokenizer_version="v1-split")

    # ------------------------------------------------------------------- i/o

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.kind,
            "tokenizer_version": self.tokenizer_version,
            "doc_count": self.doc_count,
            "index": self.backend,
        }
        with gzip.open(path, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, obj: Any) -> SparseIndex | None:
        """Build a :class:`SparseIndex` from whatever was unpickled.

        Accepts both the new payload dict and a bare legacy ``rank_bm25``
        object, so an index published before this change keeps working.
        """
        if obj is None:
            return None
        if isinstance(obj, dict) and "index" in obj:
            index = cls(
                obj["index"],
                obj.get("backend", BACKEND_BM25S),
                doc_count=int(obj.get("doc_count", 0)),
                tokenizer_version=str(obj.get("tokenizer_version", "unknown")),
            )
            if index.tokenizer_version != BM25_TOKENIZER_VERSION:
                logger.warning(
                    "BM25 index was built with tokenizer %s but the code uses %s; "
                    "rebuild with scripts/rebuild_bm25_index.py to avoid degraded recall.",
                    index.tokenizer_version,
                    BM25_TOKENIZER_VERSION,
                )
            return index
        if hasattr(obj, "get_scores"):
            logger.warning(
                "Loaded a legacy rank_bm25 index (whitespace tokenizer). "
                "Run scripts/rebuild_bm25_index.py for correct and faster sparse retrieval."
            )
            return cls.from_legacy(obj)
        logger.warning("Unrecognised BM25 payload of type %s", type(obj).__name__)
        return None

    # ----------------------------------------------------------------- query

    def top_n(self, query: str, n: int) -> list[tuple[int, float]]:
        """Return up to ``n`` ``(doc_index, score)`` pairs, best first."""
        tokens = tokenize_bm25(query)
        if not tokens or self.doc_count == 0:
            return []
        limit = max(1, min(n, self.doc_count))

        if self.kind == BACKEND_BM25S:
            return self._top_n_bm25s(tokens, limit)
        return self._top_n_legacy(tokens, limit)

    def _top_n_bm25s(self, tokens: list[str], limit: int) -> list[tuple[int, float]]:
        indices, scores = self.backend.retrieve([tokens], k=limit, show_progress=False)
        pairs = [
            (int(idx), float(score))
            for idx, score in zip(indices[0].tolist(), scores[0].tolist(), strict=False)
            if score > 0.0
        ]
        return pairs

    def _top_n_legacy(self, tokens: list[str], limit: int) -> list[tuple[int, float]]:
        scores = self.backend.get_scores(tokens)
        if scores is None:
            return []
        values = [float(value) for value in list(scores)]
        if not values:
            return []
        order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)[:limit]
        return [(int(idx), values[idx]) for idx in order if values[idx] > 0.0]
