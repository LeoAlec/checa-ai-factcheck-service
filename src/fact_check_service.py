"""Evidence-first claim verification service."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from src import observability
from src.config import (
    COLLECTION_NAME,
    CONTEXT_SCORE_MARGIN,
    EMBED_MODEL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_CONTEXT_CHARS,
    MAX_DOCUMENTS,
    RETRIEVAL_OVERFETCH,
    RETRIEVAL_TOP_K,
    SAFE_INJECTION_REFUSAL_MESSAGE,
    SAFE_REFUSAL_MESSAGE,
    SIMILARITY_THRESHOLD,
)
from src.models import (
    ClaimAnalysis,
    EvidenceConclusion,
    EvidenceLevel,
    EvidenceSource,
    FactCheckResponse,
    OperationalStatus,
    SourceOrigin,
)
from src.rag.corpus_version import corpus_version
from src.rag.embeddings import get_embedding_provider
from src.rag.guardrails import (
    CLAIM_RESPONSE_FORMAT,
    RESPONSE_FORMAT,
    enforce_grounding,
    parse_answer,
    parse_claim_analysis,
)
from src.rag.prompts import (
    ABSTENTION_MESSAGE as _ABSTENTION_MESSAGE,
    CLAIM_ANALYSIS_SYSTEM_PROMPT,
    CLAIM_ANALYSIS_USER_PROMPT,
    PROMPT_VERSION,
    PromptBundle,
    current_prompt,
    load_prompt,
)
from src.rag.retriever import (
    Evidence,
    build_context,
    evidence_from_hit,
    select_evidences,
    unique_sources,
    with_cluster_rerank,
)
from src.rag.web_search import perform_web_search, validate_web_results
from src.safety.injection import check_injection
from src.safety.moderation import check_moderation
from src.safety.validator import validate_output

logger = logging.getLogger("checa-ai.fact-check")


def _evidence_to_dict(ev: Evidence) -> dict[str, Any]:
    return {
        "titulo": ev.titulo,
        "texto_completo": ev.texto,
        "dominio": ev.dominio,
        "url": ev.url,
        "data_publicacao": ev.data_publicacao,
    }


def _local_source(ev: Evidence) -> EvidenceSource:
    domain = ev.dominio or (urlsplit(ev.url).hostname or "").removeprefix("www.")
    return EvidenceSource(
        title=ev.titulo or domain or "Evidência local",
        domain=domain,
        url=ev.url,
        relevant_excerpt=ev.texto[:2_000] or "Trecho não disponível.",
        origin=SourceOrigin.LOCAL,
        published_at=ev.data_publicacao or None,
    )


def _source_to_dict(ev: Evidence) -> dict[str, Any]:
    """Public fields plus legacy retrieval metadata for internal callers."""
    source = _local_source(ev).model_dump(mode="json")
    source.update(
        score=round(ev.score, 4),
        rerank_score=round(ev.rerank_score, 4) if ev.rerank_score is not None else None,
        cluster_id=ev.cluster_id,
        cluster_label=ev.cluster_label,
        cluster_type=ev.cluster_type,
        cluster_confidence=ev.cluster_confidence,
        cluster_review_status=ev.cluster_review_status,
    )
    return source


def _context_from_sources(sources: list[EvidenceSource], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for index, source in enumerate(sources, 1):
        block = (
            f"[{index}] Título: {source.title}\n"
            f"Domínio: {source.domain}\nURL: {source.url}\n"
            f"Origem: {source.origin.value}\nTrecho: {source.relevant_excerpt}"
        )
        if blocks and used + len(block) + 2 > max_chars:
            break
        if len(block) > max_chars:
            block = block[: max(0, max_chars - 5)].rstrip() + " [...]"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def _is_standalone_url(value: str) -> bool:
    stripped = value.strip()
    if any(char.isspace() for char in stripped):
        return False
    parsed = urlsplit(stripped)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


class FactCheckService:
    """Orchestrates safety, local retrieval, web fallback and grounded synthesis."""

    _instance: "FactCheckService | None" = None
    _llm_client: Any | None = None

    def __init__(self) -> None:
        from src.rag.qdrant import get_qdrant_client

        self._embedder = get_embedding_provider()
        self._qdrant = get_qdrant_client()

    @classmethod
    def get_instance(cls) -> "FactCheckService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _embed(self, text: str) -> list[float]:
        with observability.observation(
            "embed-query", as_type="embedding", model=EMBED_MODEL, input=text
        ) as obs:
            vector = self._embedder.embed_query(text)
            tokens = getattr(self._embedder, "last_usage_tokens", None)
            obs.update(
                usage_details={"input": tokens} if tokens else None,
                metadata={
                    "provider": getattr(self._embedder, "name", "?"),
                    "dimension": len(vector),
                },
            )
        return vector

    def _search(self, query: str, limit: int) -> list[Any]:
        return self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=self._embed(query),
            limit=limit,
            with_payload=True,
        ).points

    def _retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Evidence]:
        with observability.observation(
            "retrieval",
            as_type="retriever",
            input=query,
            metadata={
                "collection": COLLECTION_NAME,
                "top_k": top_k,
                "overfetch": RETRIEVAL_OVERFETCH,
                "max_documents": MAX_DOCUMENTS,
                "threshold": SIMILARITY_THRESHOLD,
            },
        ) as obs:
            hits = self._search(query, limit=top_k * RETRIEVAL_OVERFETCH)
            candidates = [evidence_from_hit(hit) for hit in hits]
            blocked = [item for item in candidates if item.injection_risk == "high"]
            if blocked:
                logger.warning("Evidências locais com prompt injection descartadas: %d", len(blocked))
            valid: list[Evidence] = []
            for item in candidates:
                if item.injection_risk == "high":
                    continue
                try:
                    _local_source(item)
                except ValidationError:
                    logger.warning("Evidência local inválida descartada: %s", item.url or "(sem URL)")
                    continue
                valid.append(item)
            ranked = with_cluster_rerank(valid)
            selected = select_evidences(
                ranked,
                max_chunks=top_k,
                max_documents=MAX_DOCUMENTS,
                min_score=SIMILARITY_THRESHOLD - CONTEXT_SCORE_MARGIN,
            )
            obs.update(
                output=[{"url": item.url, "score": round(item.score, 4)} for item in selected],
                metadata={
                    "candidates": len(hits),
                    "selected": len(selected),
                    "injection_blocked": len(blocked),
                },
            )
        return selected

    def _get_llm_client(self) -> Any:
        if self._llm_client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY não configurada. Defina a variável de ambiente ou use .env."
                )
            self._llm_client = openai.OpenAI(
                api_key=api_key,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        return self._llm_client

    def _call_llm(self, system: str, user: str) -> str:
        return self._chat(system, user, RESPONSE_FORMAT)

    def _call_llm_text(self, system: str, user: str) -> str:
        """Compatibility seam for web synthesis; output remains strict JSON."""
        return self._chat(system, user, RESPONSE_FORMAT)

    def _call_claim_analyzer(self, system: str, user: str) -> str:
        return self._chat(system, user, CLAIM_RESPONSE_FORMAT)

    def _analyze_claim(
        self,
        user_message: str,
        url_sources: list[EvidenceSource] | None = None,
    ) -> ClaimAnalysis:
        url_context = _context_from_sources(url_sources or [], MAX_CONTEXT_CHARS)
        user_prompt = CLAIM_ANALYSIS_USER_PROMPT.replace(
            "{{message}}", user_message
        ).replace("{{url_context}}", url_context or "(não disponível)")
        raw = self._call_claim_analyzer(CLAIM_ANALYSIS_SYSTEM_PROMPT, user_prompt)
        return parse_claim_analysis(raw)

    def _chat(self, system: str, user: str, response_format: dict | None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        bundle = current_prompt.get()
        prompt_version = bundle.version if bundle else PROMPT_VERSION
        with observability.observation(
            "llm-generation",
            as_type="generation",
            model=LLM_MODEL,
            model_parameters={"temperature": LLM_TEMPERATURE},
            input=messages,
            version=prompt_version,
            **({"prompt": bundle.langfuse_prompt} if bundle and bundle.langfuse_prompt else {}),
        ) as obs:
            response = self._get_llm_client().chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=messages,
                **({"response_format": response_format} if response_format else {}),
            )
            text = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            obs.update(
                output=text,
                usage_details=(
                    {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                    if usage
                    else None
                ),
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_version": prompt_version,
                },
            )
        return text

    @staticmethod
    def _result(
        *,
        status: OperationalStatus,
        claim: str | None,
        conclusion: EvidenceConclusion,
        sufficient: bool,
        level: EvidenceLevel,
        explanation: str,
        sources: list[EvidenceSource],
        score: float | None,
        evidence: Evidence | None = None,
        guardrails: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        public = FactCheckResponse(
            status=status,
            claim=claim,
            conclusion=conclusion,
            evidence_sufficient=sufficient,
            evidence_level=level,
            explanation=explanation,
            sources=sources,
        ).model_dump(mode="json")
        # These aliases are intentionally internal/backward-compatible and are filtered by FastAPI.
        public.update(
            score=round(score, 4) if score is not None else None,
            evidence=_evidence_to_dict(evidence) if evidence else None,
            counter_narrative=public["explanation"],
        )
        if guardrails is not None:
            public["guardrails"] = guardrails
        return public

    @classmethod
    def _abstention(
        cls,
        score: float,
        evidences: list[Evidence],
        *,
        claim: str | None,
        explanation: str = _ABSTENTION_MESSAGE,
    ) -> dict[str, Any]:
        unique = unique_sources(evidences)
        sources = [_local_source(item) for item in unique]
        result = cls._result(
            status=OperationalStatus.ABSTAINED,
            claim=claim,
            conclusion=EvidenceConclusion.INCONCLUSIVE,
            sufficient=False,
            level=EvidenceLevel.LIMITED if sources else EvidenceLevel.NONE,
            explanation=explanation,
            sources=sources,
            score=score,
            evidence=evidences[0] if evidences else None,
        )
        result["sources"] = [_source_to_dict(item) for item in unique]
        return result

    @classmethod
    def _blocked(
        cls,
        message: str,
        score: float | None = None,
        *,
        claim: str | None = None,
    ) -> dict[str, Any]:
        return cls._result(
            status=OperationalStatus.BLOCKED,
            claim=claim,
            conclusion=EvidenceConclusion.INCONCLUSIVE,
            sufficient=False,
            level=EvidenceLevel.NONE,
            explanation=message,
            sources=[],
            score=score,
        )

    def _synthesize(
        self,
        query: str,
        sources: list[EvidenceSource],
        prompt: PromptBundle,
        *,
        use_web_seam: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        context = _context_from_sources(sources, MAX_CONTEXT_CHARS)
        user_prompt = prompt.render_user(query=query, contexto=context)
        raw = (
            self._call_llm_text(prompt.render_system(), user_prompt)
            if use_web_seam
            else self._call_llm(prompt.render_system(), user_prompt)
        )
        answer, structured = parse_answer(raw)
        explanation, report = enforce_grounding(
            answer,
            [(source.url, source.domain) for source in sources],
            structured,
        )
        used_urls = set(report.used_urls)
        used_sources = [source for source in sources if source.url in used_urls]
        if report.invalid_citations:
            answer.evidence_sufficient = False
            answer.conclusion = EvidenceConclusion.INCONCLUSIVE
            answer.evidence_level = EvidenceLevel.LIMITED if used_sources else EvidenceLevel.NONE
            explanation = (
                "A resposta gerada citou uma referência que não pertence às evidências "
                "recuperadas; por segurança, a conclusão foi marcada como inconclusiva."
            )
        if not used_sources and answer.evidence_sufficient:
            answer.evidence_sufficient = False
            answer.conclusion = EvidenceConclusion.INCONCLUSIVE
            answer.evidence_level = EvidenceLevel.NONE
        result = {
            "conclusion": answer.conclusion,
            "sufficient": answer.evidence_sufficient,
            "level": answer.evidence_level,
            "explanation": explanation,
            "sources": used_sources,
            "guardrails": report.to_dict(),
        }
        return result, structured

    def _deep_search_fallback(
        self,
        query: str,
        score: float,
        abstention_evidences: list[Evidence],
        prompt: PromptBundle | None = None,
        *,
        web_sources: list[EvidenceSource] | None = None,
        abstention_explanation: str = _ABSTENTION_MESSAGE,
    ) -> dict[str, Any]:
        if web_sources is None:
            raw_results = perform_web_search(query)
            web_sources = validate_web_results(raw_results)
        if not web_sources:
            return self._abstention(
                score,
                abstention_evidences,
                claim=query,
                explanation=abstention_explanation,
            )

        prompt = prompt or load_prompt()
        token = current_prompt.set(None)
        try:
            decision, _ = self._synthesize(query, web_sources, prompt, use_web_seam=True)
        except Exception as exc:
            logger.warning("Síntese do fallback web falhou: %s", exc)
            return self._abstention(
                score,
                abstention_evidences,
                claim=query,
                explanation=abstention_explanation,
            )
        finally:
            current_prompt.reset(token)

        if not validate_output(decision["explanation"]).is_safe:
            return self._blocked(
                "A resposta gerada pela busca externa não atende aos critérios de segurança.",
                claim=query,
            )
        return self._result(
            status=OperationalStatus.DEEP_SEARCHED,
            claim=query,
            conclusion=decision["conclusion"],
            sufficient=decision["sufficient"],
            level=decision["level"],
            explanation=decision["explanation"],
            sources=decision["sources"],
            score=score,
            guardrails=decision["guardrails"],
        )

    def verify_claim(self, user_message: str, *, session_id: str | None = None) -> dict[str, Any]:
        prompt = load_prompt()
        token = current_prompt.set(prompt)
        try:
            return self._verify_traced(user_message, session_id, prompt)
        finally:
            current_prompt.reset(token)

    def _verify_traced(
        self,
        user_message: str,
        session_id: str | None,
        prompt: PromptBundle,
    ) -> dict[str, Any]:
        with observability.request_trace(
            "verify-claim",
            input=user_message,
            session_id=session_id,
            version=prompt.version,
            tags=[f"prompt:{prompt.version}", f"llm:{LLM_MODEL}", f"corpus:{corpus_version()}"],
            metadata={
                "prompt_version": prompt.version,
                "prompt_source": prompt.source,
                "llm_model": LLM_MODEL,
                "embedding_model": EMBED_MODEL,
                "collection": COLLECTION_NAME,
                "corpus_version": corpus_version(),
                "retrieval_threshold": SIMILARITY_THRESHOLD,
            },
        ) as trace:
            result = self._verify(user_message, prompt)
            result["trace_id"] = trace.trace_id
            self._score_result(trace.trace_id, result)
            trace.update(
                output=result["explanation"],
                metadata={
                    "status": result["status"],
                    "conclusion": result["conclusion"],
                    "retrieval_top_score": result["score"],
                    "n_sources": len(result["sources"]),
                },
            )
        return result

    def _verify(self, user_message: str, prompt: PromptBundle) -> dict[str, Any]:
        moderation = check_moderation(user_message)
        if not moderation.is_safe:
            logger.warning("Entrada bloqueada por moderação: %s", moderation.category)
            return self._blocked(SAFE_REFUSAL_MESSAGE)

        injection = check_injection(user_message)
        if not injection.is_safe:
            logger.warning("Entrada bloqueada por prompt injection: %s", injection.pattern)
            return self._blocked(SAFE_INJECTION_REFUSAL_MESSAGE)

        prefetched_web_sources: list[EvidenceSource] | None = None
        if _is_standalone_url(user_message):
            prefetched_web_sources = validate_web_results(perform_web_search(user_message))

        claim_analysis = self._analyze_claim(user_message, prefetched_web_sources)
        if not claim_analysis.has_verifiable_claim or not claim_analysis.claim:
            explanation = claim_analysis.explanation
            if claim_analysis.missing_context:
                missing = "; ".join(claim_analysis.missing_context)
                explanation = f"{explanation} Para verificar, informe: {missing}."
            if not validate_output(explanation).is_safe:
                explanation = (
                    "Não foi possível identificar uma alegação verificável com o contexto "
                    "fornecido. Inclua o fato, a data ou o local relevante."
                )
            return self._abstention(
                0.0,
                [],
                claim=None,
                explanation=explanation,
            )

        claim = claim_analysis.claim
        evidences = self._retrieve(claim)
        if not evidences:
            return self._deep_search_fallback(
                claim,
                0.0,
                [],
                prompt,
                web_sources=prefetched_web_sources,
            )

        best_score = max(item.score for item in evidences)
        if best_score < SIMILARITY_THRESHOLD:
            return self._deep_search_fallback(
                claim,
                best_score,
                [],
                prompt,
                web_sources=prefetched_web_sources,
            )

        context, used_evidences = build_context(evidences, MAX_CONTEXT_CHARS)
        # Keep the mature grouping/truncation behavior from the local RAG path.
        local_sources = [_local_source(item) for item in unique_sources(used_evidences)]
        raw = self._call_llm(
            prompt.render_system(),
            prompt.render_user(query=claim, contexto=context),
        )
        answer, structured = parse_answer(raw)
        explanation, report = enforce_grounding(
            answer,
            [(source.url, source.domain) for source in local_sources],
            structured,
        )
        used_urls = set(report.used_urls)
        result_sources = [source for source in local_sources if source.url in used_urls]

        if report.invalid_citations:
            answer.evidence_sufficient = False
            answer.conclusion = EvidenceConclusion.INCONCLUSIVE

        if answer.conclusion is EvidenceConclusion.INCONCLUSIVE or not answer.evidence_sufficient:
            evidence_by_url = {item.url: item for item in unique_sources(used_evidences)}
            cited_evidences = [
                evidence_by_url[url]
                for url in report.used_urls
                if url in evidence_by_url
            ]
            result = self._deep_search_fallback(
                claim,
                best_score,
                cited_evidences,
                prompt,
                web_sources=prefetched_web_sources,
                abstention_explanation=(
                    explanation if cited_evidences else _ABSTENTION_MESSAGE
                ),
            )
            if result["status"] == "abstained":
                result["abstained_by"] = "model"
            return result

        if not result_sources:
            return self._deep_search_fallback(
                claim,
                best_score,
                [],
                prompt,
                web_sources=prefetched_web_sources,
            )

        if not validate_output(explanation).is_safe:
            return self._blocked(
                "A resposta gerada não atende aos critérios de segurança e não pode ser exibida.",
                best_score,
                claim=claim,
            )

        result = self._result(
            status=OperationalStatus.MATCHED,
            claim=claim,
            conclusion=answer.conclusion,
            sufficient=answer.evidence_sufficient,
            level=answer.evidence_level,
            explanation=explanation,
            sources=result_sources,
            score=best_score,
            evidence=used_evidences[0],
            guardrails=report.to_dict(),
        )
        by_url = {item.url: item for item in unique_sources(used_evidences)}
        result["sources"] = [_source_to_dict(by_url[source.url]) for source in result_sources]
        return result

    @staticmethod
    def _score_result(trace_id: str | None, result: dict[str, Any]) -> None:
        if not trace_id:
            return
        if result["score"] is not None:
            observability.score_trace(trace_id, "top_similarity", result["score"])
        for flag in ("abstained", "deep_searched", "blocked"):
            observability.score_trace(
                trace_id,
                flag,
                float(result["status"] == flag),
                data_type="BOOLEAN",
            )
        if result["status"] in {"matched", "deep_searched"} and result.get("guardrails"):
            bad = result["guardrails"]["ungrounded_urls"]
            observability.score_trace(
                trace_id,
                "sources_grounded",
                float(not bad),
                data_type="BOOLEAN",
                comment=None if not bad else f"URLs fora do contexto: {bad}"[:500],
            )
            observability.score_trace(
                trace_id,
                "structured_output",
                float(result["guardrails"]["structured"]),
                data_type="BOOLEAN",
            )
