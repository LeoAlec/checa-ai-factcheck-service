"""
src/fact_check_service.py
─────────────────────────────────────────────────────────────────────
Serviço de verificação de fatos via pipeline RAG:
  1. Moderação e Guardrails de Entrada (Toxicidade e Injection)
  2. Converte a mensagem do usuário em embedding (sentence-transformers)
  3. Busca no Qdrant os chunks mais similares (top-k) e deduplica por documento
  4. Se o melhor score >= limiar, gera contranarrativa via GPT-4o-mini
     usando múltiplas evidências numeradas
  5. Validação de Guardrails de Saída
  6. Caso contrário, retorna abstinência
"""

from __future__ import annotations

import logging
import os
import re
from contextvars import ContextVar
from urllib.parse import urlsplit

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
    SIMILARITY_THRESHOLD,
    SAFE_REFUSAL_MESSAGE,
    SAFE_INJECTION_REFUSAL_MESSAGE,
)
from src.rag.corpus_version import corpus_version
from src.rag.embeddings import get_embedding_provider
from src.rag.guardrails import RESPONSE_FORMAT, enforce_grounding, parse_answer
from src.rag.prompts import (
    ABSTENTION_MESSAGE as _ABSTENTION_MESSAGE,
    PROMPT_VERSION,
    current_prompt,
    load_prompt,
)
from src.rag.qdrant import get_qdrant_client
from src.rag.retriever import (
    Evidence,
    build_context,
    evidence_from_hit,
    select_evidences,
    unique_sources,
    with_cluster_rerank,
)
from src.rag.web_search import perform_web_search
from src.safety.moderation import check_moderation
from src.safety.injection import check_injection
from src.safety.validator import validate_output

# Configure logging for safety events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("checa-ai.safety")

_BACKEND_SYSTEM_PROMPT = """\
Você verifica alegações usando somente os laudos fornecidos. Escreva uma
contranarrativa clara, em português, de até {max_chars} caracteres para WhatsApp.
Cite cada laudo usado pelo número entre colchetes, como [1]. Não inclua
links nem uma lista de fontes: a API acrescentará as fontes citadas.
Não invente fatos ou citações.
"""

_BACKEND_USER_PROMPT = """\
Alegação: {query}

Laudos de checagem:
{contexto}

Responda apenas com a contranarrativa curta e suas citações numeradas.
"""

_NARRATIVE_LIMIT = 500
_REPLY_FIXED_RESERVE = 350
_MIN_NARRATIVE_CHARS = 80
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CHUNK_CITATION_PATTERN = re.compile(r"\[(\d+),\s*trecho\s+\d+\]", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"'<>]+")
_generation_version: ContextVar[str | None] = ContextVar("generation_version", default=None)


class VerificationGenerationError(RuntimeError):
    """A geração não produziu um resultado fundamentado para o contrato v1."""


def _safe_source_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not any(char.isspace() for char in url)
        )
    except (TypeError, ValueError):
        return False


def _evidence_to_dict(ev: Evidence) -> dict:
    return {
        "titulo": ev.titulo,
        "texto_completo": ev.texto,
        "dominio": ev.dominio,
        "url": ev.url,
        "data_publicacao": ev.data_publicacao,
    }


def _source_to_dict(ev: Evidence) -> dict:
    return {
        "title": ev.titulo,
        "domain": ev.dominio,
        "url": ev.url,
        "published_at": ev.data_publicacao,
        "score": round(ev.score, 4),
        "rerank_score": round(ev.rerank_score, 4) if ev.rerank_score is not None else None,
        "cluster_id": ev.cluster_id,
        "cluster_label": ev.cluster_label,
        "cluster_type": ev.cluster_type,
        "cluster_confidence": ev.cluster_confidence,
        "cluster_review_status": ev.cluster_review_status,
    }


class FactCheckService:
    """
    Singleton que carrega modelo e cliente Qdrant uma única vez.
    Use `FactCheckService.get_instance()` em vez de instanciar diretamente.
    """

    _instance: "FactCheckService | None" = None

    def __init__(self) -> None:
        self._embedder = get_embedding_provider()
        self._qdrant = get_qdrant_client()
        self._llm_client = None

    @classmethod
    def get_instance(cls) -> "FactCheckService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ─── Embedding ────────────────────────────────────────────────────
    def _embed(self, text: str) -> list[float]:
        with observability.observation(
            "embed-query", as_type="embedding", model=EMBED_MODEL, input=text
        ) as obs:
            vec = self._embedder.embed_query(text)
            tokens = getattr(self._embedder, "last_usage_tokens", None)
            obs.update(
                usage_details={"input": tokens} if tokens else None,
                metadata={"provider": getattr(self._embedder, "name", "?"), "dimension": len(vec)},
            )
        return vec

    # ─── Retrieval ────────────────────────────────────────────────────
    def _search(self, query: str, limit: int):
        vec = self._embed(query)
        return self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=limit,
            with_payload=True,
        ).points

    def _retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Evidence]:
        """Busca candidatos, deduplica por documento e limita chunks/documentos."""
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
            candidates = [evidence_from_hit(h) for h in hits]
            # Guardrail: laudos raspados da web são entrada não confiável.
            blocked = [ev for ev in candidates if ev.injection_risk == "high"]
            if blocked:
                logger.warning(
                    "Guardrail: %d chunk(s) com prompt injection excluídos",
                    len(blocked),
                )
            evidences = with_cluster_rerank([ev for ev in candidates if ev.injection_risk != "high"])
            selected = select_evidences(
                evidences,
                max_chunks=top_k,
                max_documents=MAX_DOCUMENTS,
                min_score=SIMILARITY_THRESHOLD - CONTEXT_SCORE_MARGIN,
            )
            obs.update(
                output=[
                    {"url": ev.url, "score": round(ev.score, 4), "cluster": ev.cluster_label}
                    for ev in selected
                ],
                metadata={
                    "candidates": len(hits),
                    "selected": len(selected),
                    "injection_blocked": len(blocked),
                    "injection_medium": sum(ev.injection_risk == "medium" for ev in selected),
                },
            )
        return selected

    # ─── LLM ─────────────────────────────────────────────────────────
    _llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY não configurada. "
                    "Defina a variável de ambiente ou adicione ao arquivo .env."
                )
            # Um cliente por processo (pool de conexões), com timeout e retry.
            self._llm_client = openai.OpenAI(
                api_key=api_key, timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES
            )
        return self._llm_client

    def _call_llm(self, system: str, user: str) -> str:
        """Geração do debunk: saída estruturada {texto, fontes_usadas}."""
        return self._chat(system, user, RESPONSE_FORMAT)

    def _call_llm_text(self, system: str, user: str) -> str:
        """Texto livre (síntese do DeepSearch, que tem prompt próprio)."""
        return self._chat(system, user, None)

    def _chat(self, system: str, user: str, response_format: dict | None) -> str:
        client = self._get_llm_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        bundle = current_prompt.get()
        prompt_version = _generation_version.get() or (bundle.version if bundle else PROMPT_VERSION)
        with observability.observation(
            "llm-generation",
            as_type="generation",
            model=LLM_MODEL,
            model_parameters={"temperature": LLM_TEMPERATURE},
            input=messages,
            version=prompt_version,
            **({"prompt": bundle.langfuse_prompt} if bundle and bundle.langfuse_prompt else {}),
        ) as obs:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=messages,
                **({"response_format": response_format} if response_format else {}),
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            obs.update(
                output=text,
                # Langfuse calcula o custo a partir de model + usage_details.
                usage_details=(
                    {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                    if usage
                    else None
                ),
                metadata={"finish_reason": resp.choices[0].finish_reason, "prompt_version": prompt_version},
            )
        return text

    def _deep_search_fallback(self, query: str, score: float, evidences: list[Evidence]) -> dict:
        """
        Fallback quando não há match no RAG: pesquisa na web e sintetiza a resposta.
        """
        from src.config import DEEP_SEARCH_SYSTEM_PROMPT, DEEP_SEARCH_USER_PROMPT

        logger.info("Iniciando DeepSearch")

        # 1. Busca Web
        web_results = perform_web_search(query)
        logger.info("Resultados da web encontrados: %d", len(web_results))

        if not web_results:
            # Se nem a web retornou nada, voltamos para a abstenção clássica
            return self._abstention(score, evidences)

        # Formata resultados para o prompt
        formatted_results = "\n\n".join([
            f"Fonte: {r['title']} ({r['url']})\nConteúdo: {r['snippet']}"
            for r in web_results
        ])

        user_prompt = DEEP_SEARCH_USER_PROMPT.format(
            query=query,
            web_results=formatted_results
        )

        # 2. Geração de síntese cautelosa
        # Prompt do DeepSearch é outro: não atribuir a versão do prompt "debunk" a esta generation.
        token = current_prompt.set(None)
        try:
            narrative = self._call_llm_text(DEEP_SEARCH_SYSTEM_PROMPT, user_prompt)
        except Exception:
            logger.error("Erro ao gerar síntese DeepSearch")
            return self._abstention(score, evidences)
        finally:
            current_prompt.reset(token)

        # 3. Validação de Segurança (Saída)
        val_result = validate_output(narrative)
        if not val_result.is_safe:
            logger.error("Blocked harmful deep search output: category=%s", val_result.category)
            return {
                "status": "blocked",
                "score": None,
                "evidence": None,
                "sources": [],
                "counter_narrative": "Desculpe, mas a pesquisa web gerou um conteúdo que não atende aos nossos critérios de segurança.",
            }

        return {
            "status": "deep_searched",
            "score": round(score, 4),
            "evidence": None,
            "sources": [], # Em um sistema real, poderíamos adicionar as URLs da web aqui
            "counter_narrative": narrative,
        }

    # ─── Public API ───────────────────────────────────────────────────
    @staticmethod
    def _abstention(score: float, evidences: list[Evidence]) -> dict:
        sources = unique_sources(evidences)
        return {
            "status": "abstained",
            "score": round(score, 4),
            "evidence": _evidence_to_dict(evidences[0]) if evidences else None,
            "sources": [_source_to_dict(s) for s in sources],
            "counter_narrative": _ABSTENTION_MESSAGE,
        }

    def verify_claim(self, user_message: str, *, session_id: str | None = None) -> dict:
        """
        Executa o pipeline RAG completo com camadas de segurança, dentro de um
        trace do Langfuse (inclusive as requisições bloqueadas).

        Retorno:
            {
                "status":            "matched" | "abstained" | "deep_searched" | "blocked",
                "score":             float | None,
                "evidence":          dict | None,
                "sources":           list[dict],
                "counter_narrative": str,
                "trace_id":          str | None,       # id p/ feedback; None sem Langfuse
            }
        """
        prompt = load_prompt()
        token = current_prompt.set(prompt)
        try:
            return self._verify_traced(user_message, session_id, prompt)
        finally:
            current_prompt.reset(token)

    def _verify_traced(self, user_message: str, session_id: str | None, prompt) -> dict:
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
                "threshold": SIMILARITY_THRESHOLD,
            },
        ) as trace:
            result = self._verify(user_message, prompt)
            result["trace_id"] = trace.trace_id
            self._score_result(trace.trace_id, result)
            trace.update(
                output=result["counter_narrative"],
                metadata={
                    "status": result["status"],
                    "top_score": result["score"],
                    "n_sources": len(result["sources"]),
                },
            )
        return result

    def _verify(self, user_message: str, prompt) -> dict:
        # 1. Input Guardrail: Moderation (Toxicity)
        mod_result = check_moderation(user_message)
        if not mod_result.is_safe:
            logger.warning(f"Blocked toxic input: category={mod_result.category}")
            return {
                "status": "blocked",
                "score": None,
                "evidence": None,
                "sources": [],
                "counter_narrative": SAFE_REFUSAL_MESSAGE,
            }

        # 2. Input Guardrail: Prompt Injection
        inj_result = check_injection(user_message)
        if not inj_result.is_safe:
            logger.warning(f"Blocked injection attempt: pattern={inj_result.pattern}")
            return {
                "status": "blocked",
                "score": None,
                "evidence": None,
                "sources": [],
                "counter_narrative": SAFE_INJECTION_REFUSAL_MESSAGE,
            }

        # 3. RAG Retrieval
        evidences = self._retrieve(user_message)

        if not evidences:
            return self._deep_search_fallback(user_message, 0.0, [])

        best_score = max(ev.score for ev in evidences)
        if best_score < SIMILARITY_THRESHOLD:
            return self._deep_search_fallback(user_message, best_score, evidences)

        # 4. Generation
        contexto, used = build_context(evidences, MAX_CONTEXT_CHARS)
        user_prompt = prompt.render_user(query=user_message, contexto=contexto)

        raw = self._call_llm(prompt.render_system(), user_prompt)
        sources = unique_sources(used)
        answer, structured = parse_answer(raw)
        if not answer.laudos_tratam_alegacao:
            # O próprio modelo sinalizou que os laudos não cobrem a alegação:
            # abstém-se em vez de responder de memória e creditar fonte irrelevante.
            logger.info("Guardrail: modelo se absteve (laudos não tratam da alegação).")
            # Mesmo caminho do score baixo: DeepSearch, ou abstenção se a web não ajudar.
            result = self._deep_search_fallback(user_message, best_score, used)
            if result["status"] == "abstained":
                result["abstained_by"] = "model"
            return result
        counter_narrative, report = enforce_grounding(
            answer, [(s.url, s.dominio) for s in sources], structured
        )
        if report.sanitized:
            logger.warning("Guardrail: URLs fora do contexto removidas: %s", report.ungrounded_urls)

        # 5. Output Guardrail: Safety Validation
        val_result = validate_output(counter_narrative)
        if not val_result.is_safe:
            logger.error(f"Blocked harmful output: category={val_result.category}")
            # In case of harmful output, we return a fallback refusal
            return {
                "status": "blocked",
                "score": round(best_score, 4),
                "evidence": _evidence_to_dict(used[0]),
                "sources": [_source_to_dict(s) for s in unique_sources(used)],
                "counter_narrative": "Desculpe, mas a resposta gerada não atende aos nossos critérios de segurança e não pode ser exibida.",
            }

        return {
            "status": "matched",
            "score": round(best_score, 4),
            "evidence": _evidence_to_dict(used[0]),
            "sources": [_source_to_dict(s) for s in sources],
            "counter_narrative": counter_narrative,
            "guardrails": report.to_dict(),
        }

    def verify_claim_for_backend(self, text: str) -> dict:
        """Return the grounded, compact v1 result consumed by the backend."""
        with observability.request_trace(
            "verify-v1",
            input=text,
            version="verify-v1",
            tags=["prompt:verify-v1", f"llm:{LLM_MODEL}", f"corpus:{corpus_version()}"],
            metadata={"collection": COLLECTION_NAME, "threshold": SIMILARITY_THRESHOLD},
        ) as trace:
            result = self._verify_claim_for_backend(text)
            trace.update(
                output=result["counter_narrative"],
                metadata={"verdict": result["verdict"], "top_score": result["similarity_score"]},
            )
            return result

    def _verify_claim_for_backend(self, text: str) -> dict:
        mod_result = check_moderation(text)
        if not mod_result.is_safe:
            logger.warning("Blocked toxic v1 input: category=%s", mod_result.category)
            return self._backend_result("blocked", None, SAFE_REFUSAL_MESSAGE)
        inj_result = check_injection(text)
        if not inj_result.is_safe:
            logger.warning("Blocked v1 injection: pattern=%s", inj_result.pattern)
            return self._backend_result("blocked", None, SAFE_INJECTION_REFUSAL_MESSAGE)

        evidences = self._retrieve(text)
        best_score = max((ev.score for ev in evidences), default=0.0)
        result = self._backend_result("insufficient_evidence", best_score)
        if best_score < SIMILARITY_THRESHOLD:
            return self._backend_fallback(text, best_score, evidences)

        selected_sources: list[Evidence] = []
        selected_urls: set[str] = set()
        source_chars = 0
        max_source_chars = 1000 - _REPLY_FIXED_RESERVE - _MIN_NARRATIVE_CHARS
        for ev in unique_sources(evidences):
            if not _safe_source_url(ev.url):
                continue
            additional = 0 if ev.url in selected_urls else len(ev.url) + bool(selected_urls)
            if source_chars + additional > max_source_chars:
                continue
            selected_sources.append(ev)
            selected_urls.add(ev.url)
            source_chars += additional
            if len(selected_sources) == 2:
                break
        if not selected_sources or not any(
            ev.score >= SIMILARITY_THRESHOLD for ev in selected_sources
        ):
            raise VerificationGenerationError("nenhuma fonte válida cabe na resposta")

        narrative_limit = min(
            _NARRATIVE_LIMIT, 1000 - _REPLY_FIXED_RESERVE - source_chars
        )
        selected_ids = {ev.document_id for ev in selected_sources}
        selected = [ev for ev in evidences if ev.document_id in selected_ids]
        context, used = build_context(selected, MAX_CONTEXT_CHARS)
        prompt = _BACKEND_USER_PROMPT.format(query=text, contexto=context)
        system_prompt = _BACKEND_SYSTEM_PROMPT.format(max_chars=narrative_limit)
        token = _generation_version.set("verify-v1")
        try:
            raw = self._call_llm(system_prompt, prompt)
        finally:
            _generation_version.reset(token)
        answer, _ = parse_answer(raw)
        if not answer.laudos_tratam_alegacao:
            return self._backend_fallback(text, best_score, used)
        narrative = answer.texto.strip()
        if len(narrative) > narrative_limit:
            condensation_prompt = (
                f"Reduza o texto a no máximo {narrative_limit} caracteres, "
                "mantendo somente fatos dos laudos e as citações numeradas. "
                f"Responda apenas com o texto reduzido.\n\n{narrative}"
            )
            token = _generation_version.set("verify-v1")
            try:
                raw = self._call_llm(system_prompt, condensation_prompt)
            finally:
                _generation_version.reset(token)
            answer, _ = parse_answer(raw)
            if not answer.laudos_tratam_alegacao:
                return self._backend_fallback(text, best_score, used)
            narrative = answer.texto.strip()
        narrative = _CHUNK_CITATION_PATTERN.sub(r"[\1]", narrative)
        if len(narrative) > narrative_limit:
            raise VerificationGenerationError("contranarrativa excede o tamanho permitido")

        cited = {int(number) for number in _CITATION_PATTERN.findall(narrative)}
        if not cited or any(number < 1 or number > len(used) for number in cited):
            raise VerificationGenerationError("citação sem fonte recuperada")

        sources = []
        seen_urls: set[str] = set()
        for number, ev in enumerate(used, 1):
            if number not in cited:
                continue
            if not _safe_source_url(ev.url):
                raise VerificationGenerationError("fonte citada possui URL inválida")
            if ev.url not in seen_urls:
                sources.append({"title": ev.titulo, "domain": ev.dominio, "url": ev.url})
                seen_urls.add(ev.url)
        allowed_urls = seen_urls
        narrative_urls = {url.rstrip(".,;:!?") for url in _URL_PATTERN.findall(narrative)}
        if narrative_urls - allowed_urls or set(answer.fontes_usadas) - allowed_urls:
            raise VerificationGenerationError("URL sem fonte recuperada")

        val_result = validate_output(narrative)
        if not val_result.is_safe:
            logger.error("Blocked harmful v1 output: category=%s", val_result.category)
            return self._backend_result(
                "blocked", best_score,
                "Desculpe, mas a resposta gerada não atende aos nossos critérios de segurança e não pode ser exibida.",
            )

        result.update(verdict="matched", counter_narrative=narrative, sources=sources)
        return result

    @staticmethod
    def _backend_result(verdict: str, score: float | None, narrative: str | None = None) -> dict:
        return {
            "schema_version": 1,
            "verdict": verdict,
            "similarity_score": round(score, 4) if score is not None else None,
            "counter_narrative": narrative,
            "sources": [],
        }

    def _backend_fallback(self, text: str, score: float, evidences: list[Evidence]) -> dict:
        result = self._deep_search_fallback(text, score, evidences)
        if result["status"] == "deep_searched":
            return self._backend_result("deep_searched", score, result["counter_narrative"])
        if result["status"] == "blocked":
            return self._backend_result("blocked", result["score"], result["counter_narrative"])
        return self._backend_result("insufficient_evidence", score)

    @staticmethod
    def _score_result(trace_id: str | None, result: dict) -> None:
        """Scores online automáticos: viram séries/filtros no dashboard do Langfuse."""
        if not trace_id:
            return
        if result["score"] is not None:  # bloqueios de entrada não têm score
            observability.score_trace(trace_id, "top_similarity", result["score"])
        for flag in ("abstained", "deep_searched", "blocked"):
            observability.score_trace(
                trace_id, flag, float(result["status"] == flag), data_type="BOOLEAN"
            )
        if result["status"] == "matched":
            # Mede a saída crua do LLM (antes da sanitização): taxa de link inventado.
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
