from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from egov_bot.config import Settings
from egov_bot.conversation.followup_detector import is_followup
from egov_bot.conversation.session_manager import SessionManager
from egov_bot.data.procedure_store import ProcedureStore
from egov_bot.rag.answer_generator import GeminiAnswerGenerator
from egov_bot.rag.citation import append_sources_if_missing, unique_sources
from egov_bot.rag.guardrails import looks_in_domain, out_of_domain_answer
from egov_bot.rag.prompt_builder import build_fallback_answer, build_prompt
from egov_bot.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from egov_bot.schemas.common import ChatResult, Source
from egov_bot.storage.repositories import AppRepository
from egov_bot.utils.cache import TTLCache
from egov_bot.utils.normalizer import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class ContextBundle:
    context: str
    parent_id: str | None
    sources: list[Source]
    retrieval_results: list[RetrievalResult]


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        procedure_store: ProcedureStore,
        retriever: HybridRetriever,
        sessions: SessionManager,
        repository: AppRepository,
        generator: GeminiAnswerGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.procedure_store = procedure_store
        self.retriever = retriever
        self.sessions = sessions
        self.repository = repository
        self.generator = generator or GeminiAnswerGenerator(settings)
        self.cache: TTLCache[str, ChatResult] = TTLCache(
            max_items=settings.answer_cache_max_items,
            ttl_seconds=settings.answer_cache_ttl_seconds,
        )

    def answer(self, question: str, session_id: str, request_id: str) -> ChatResult:
        start = time.perf_counter()
        question = question.strip()
        cache_key = self._cache_key(question, session_id)
        cached = self.cache.get(cache_key)
        if cached is not None:
            result = ChatResult(
                answer=cached.answer,
                sources=cached.sources,
                request_id=request_id,
                latency_ms=int((time.perf_counter() - start) * 1000),
                cached=True,
                context_source=cached.context_source,
                timings=cached.timings,
            )
            self.repository.log_query(
                request_id,
                session_id,
                question,
                result.answer,
                result.sources,
                result.latency_ms,
                cached=True,
            )
            return result

        timings: dict[str, int] = {}
        retrieval_start = time.perf_counter()
        context = self._context_for(question, session_id)
        timings["retrieval_ms"] = int((time.perf_counter() - retrieval_start) * 1000)

        if not looks_in_domain(question, context.sources):
            answer = out_of_domain_answer()
        elif self.generator.available:
            generation_start = time.perf_counter()
            prompt = build_prompt(
                history=self.sessions.history_text(session_id),
                context=context.context,
                question=question,
                sources=context.sources,
            )
            try:
                answer = self.generator.generate(prompt)
            except Exception as exc:
                logger.warning("Answer generation failed; using extractive fallback: %s", exc)
                answer = build_fallback_answer(context.context, context.sources)
            timings["generation_ms"] = int((time.perf_counter() - generation_start) * 1000)
            answer = append_sources_if_missing(answer, context.sources)
        else:
            answer = build_fallback_answer(context.context, context.sources)

        result = ChatResult(
            answer=answer,
            sources=context.sources,
            request_id=request_id,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cached=False,
            context_source=context.parent_id,
            timings=timings,
        )
        self.cache.set(cache_key, result)
        self.sessions.append_turn(
            session_id=session_id,
            question=question,
            answer=answer,
            context=context.context,
            parent_id=context.parent_id,
            sources=context.sources,
        )
        self._update_popular(context.sources)
        self.repository.log_query(
            request_id,
            session_id,
            question,
            answer,
            context.sources,
            result.latency_ms,
            cached=False,
        )
        return result

    def _context_for(self, question: str, session_id: str) -> ContextBundle:
        previous = self.sessions.last_context(session_id)
        if previous and is_followup(question, has_history=True):
            return ContextBundle(
                context=previous.context,
                parent_id=previous.parent_id,
                sources=previous.sources,
                retrieval_results=[],
            )

        results = self.retriever.retrieve(question, top_k=self.settings.top_k)
        sources = unique_sources([result.to_source(self.procedure_store) for result in results], limit=5)
        context_parts: list[str] = []
        seen_parent_ids: set[str] = set()
        for result in results:
            if not result.parent_id or result.parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(result.parent_id)
            full_text = self.procedure_store.format_procedure(result.parent_id)
            context_parts.append(full_text or result.text)

        return ContextBundle(
            context="\n\n---\n\n".join(part for part in context_parts if part),
            parent_id=results[0].parent_id if results else None,
            sources=sources,
            retrieval_results=results,
        )

    def _update_popular(self, sources: list[Source]) -> None:
        for source in sources[:1]:
            self.repository.increment_popular(source.title, source.url)

    def _cache_key(self, question: str, session_id: str) -> str:
        last = self.sessions.last_context(session_id)
        parent_id = last.parent_id if last else ""
        raw = "|".join(
            [
                normalize_text(question),
                session_id,
                parent_id or "",
                self.settings.emb_model,
                self.settings.genai_model,
                str(self.settings.top_k),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
