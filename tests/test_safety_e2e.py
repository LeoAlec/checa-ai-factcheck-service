"""Smoke E2E against real OpenAI, Tavily and Qdrant services."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import app as app_module
from src import fact_check_service as fcs
from src.config import COLLECTION_NAME, EMBED_MODEL
from src.fact_check_service import FactCheckService
from src.models import FactCheckResponse
from src.rag import web_search


pytestmark = [
    pytest.mark.live_e2e,
    pytest.mark.skipif(not os.environ.get("RUN_E2E"), reason="defina RUN_E2E=1"),
]

LOCAL_SUPPORTED_CLAIM = os.environ.get(
    "E2E_LOCAL_SUPPORTED_CLAIM",
    "A Prefeitura de Cidade Exemplo inaugurou o Hospital Horizonte em 15 de março de 2024.",
)
WEB_CONTRADICTED_CLAIM = os.environ.get(
    "E2E_WEB_CONTRADICTED_CLAIM",
    (
        "A Organização Mundial da Saúde declarou em 2024 que as vacinas "
        "contra a covid-19 contêm microchips."
    ),
)
STANDALONE_URL = os.environ.get(
    "E2E_STANDALONE_URL",
    (
        "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/c/covid-19/"
        "perguntas-frequentes/faq/vacina/a-vacina-contra-a-covid-19"
    ),
)
EXTERNAL_FAILURE_CLAIM = os.environ.get(
    "E2E_EXTERNAL_FAILURE_CLAIM",
    (
        "A Prefeitura de Atlântida inaugurou um metrô submarino "
        "em 31 de dezembro de 2099."
    ),
)


def _required_environment() -> None:
    required = ("OPENAI_API_KEY", "TAVILY_API_KEY", "QDRANT_URL", "QDRANT_COLLECTION")
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    assert not missing, "variáveis obrigatórias ausentes: " + ", ".join(missing)


def _vector_size(collection_info: Any) -> int:
    vectors = collection_info.config.params.vectors
    assert not isinstance(vectors, dict), "o smoke E2E requer uma coleção com vetor sem nome"
    return int(vectors.size)


@pytest.fixture(scope="module")
def live_client() -> Generator[TestClient, None, None]:
    _required_environment()
    qdrant_url = os.environ["QDRANT_URL"].strip()
    assert qdrant_url.startswith(("http://", "https://")), (
        "QDRANT_URL deve apontar para o servidor HTTP; fallback local não é aceito no E2E"
    )

    qdrant = QdrantClient(url=qdrant_url, timeout=5)
    info = qdrant.get_collection(COLLECTION_NAME)
    assert (info.points_count or 0) > 0, f"coleção {COLLECTION_NAME!r} está vazia"

    FactCheckService._instance = None
    service = FactCheckService.get_instance()
    assert _vector_size(info) == service._embedder.dimension, (
        "dimensão da coleção incompatível com o embedding configurado"
    )

    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    indexed_models = {
        str((point.payload or {}).get("embedding_model") or "") for point in points
    }
    assert EMBED_MODEL in indexed_models, (
        "nenhum ponto amostrado foi indexado com o EMBEDDING_MODEL configurado"
    )

    with TestClient(app_module.app) as client:
        yield client

    FactCheckService._instance = None


def _verify(client: TestClient, message: str) -> FactCheckResponse:
    response = client.post("/api/debunk", json={"message": message})
    assert response.status_code == 200, response.text
    return FactCheckResponse.model_validate(response.json())


def test_live_local_evidence_supports_claim(live_client):
    result = _verify(live_client, LOCAL_SUPPORTED_CLAIM)

    assert result.status.value == "matched"
    assert result.conclusion.value == "sustentada"
    assert result.evidence_sufficient is True
    assert result.sources and all(source.origin.value == "local" for source in result.sources)
    assert result.probabilistic_assessment is None


def test_live_web_fallback_contradicts_claim(live_client):
    result = _verify(live_client, WEB_CONTRADICTED_CLAIM)

    assert result.status.value == "deep_searched"
    assert result.conclusion.value == "contradita"
    assert result.evidence_sufficient is True
    assert result.sources and all(source.origin.value == "web" for source in result.sources)


def test_live_vague_message_abstains_without_sources(live_client):
    result = _verify(live_client, "Isso é verdade?")

    assert result.status.value == "abstained"
    assert result.claim is None
    assert result.conclusion.value == "inconclusiva"
    assert result.evidence_sufficient is False
    assert result.sources == []


def test_live_standalone_url_is_resolved(live_client):
    result = _verify(live_client, STANDALONE_URL)

    assert result.claim
    assert result.status.value == "deep_searched"
    assert result.sources and all(source.origin.value == "web" for source in result.sources)


def test_live_external_search_failure_abstains(live_client, monkeypatch):
    calls: list[str] = []
    real_search = web_search.perform_web_search

    def failing_provider_call(query: str, max_results: int = 5):
        calls.append(query)
        return real_search(query, max_results=max_results)

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-invalid-e2e-credential")
    monkeypatch.setattr(fcs, "perform_web_search", failing_provider_call)

    result = _verify(live_client, EXTERNAL_FAILURE_CLAIM)

    assert calls, "o fallback externo não foi exercitado"
    assert result.status.value == "abstained"
    assert result.conclusion.value == "inconclusiva"
    assert result.evidence_sufficient is False
    assert result.sources == []
