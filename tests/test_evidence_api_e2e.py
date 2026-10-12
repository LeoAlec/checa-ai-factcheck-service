"""Hermetic HTTP end-to-end tests for the public evidence flow."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as app_module
from src import fact_check_service as fcs


LOCAL_URL = "https://checamos.afp.com/e2e-local"
WEB_URL = "https://www.gov.br/saude/e2e-web"


def _decision(conclusion: str, *, url: str, sufficient: bool = True) -> str:
    return json.dumps(
        {
            "conclusion": conclusion,
            "evidence_sufficient": sufficient,
            "evidence_level": "forte" if sufficient else "limitada",
            "explanation": (
                "A fonte resolve a alegação [1]."
                if sufficient
                else "A fonte trata de outro fato [1]."
            ),
            "source_urls": [url],
        },
        ensure_ascii=False,
    )


def _claim_analysis(
    claim: str | None = "O ministério confirmou a medida em Recife em 2026",
) -> str:
    return json.dumps(
        {
            "has_verifiable_claim": claim is not None,
            "claim": claim,
            "missing_context": [] if claim else ["qual medida", "data e local"],
            "explanation": (
                "Alegação factual identificada."
                if claim
                else "A mensagem não contém um fato específico."
            ),
        },
        ensure_ascii=False,
    )


def _hit(text: str = "O documento oficial confirma a medida."):
    return SimpleNamespace(
        score=0.93,
        payload={
            "document_id": "e2e-local",
            "chunk_index": 0,
            "texto_chunk": text,
            "titulo": "Checagem local",
            "dominio": "checamos.afp.com",
            "url": LOCAL_URL,
            "data_publicacao": "2026-05-11",
        },
    )


@pytest.fixture
def http_flow(monkeypatch):
    service = object.__new__(fcs.FactCheckService)
    service.hits = [_hit()]
    service.claim_answer = _claim_analysis()
    service.local_answer = _decision("sustentada", url=LOCAL_URL)
    service.web_answer = _decision("contradita", url=WEB_URL)
    service.search_queries = []

    monkeypatch.setattr(
        service,
        "_search",
        lambda query, limit: service.search_queries.append(query) or service.hits,
        raising=False,
    )
    monkeypatch.setattr(
        service,
        "_call_claim_analyzer",
        lambda system, user: service.claim_answer,
        raising=False,
    )
    monkeypatch.setattr(
        service, "_call_llm", lambda system, user: service.local_answer, raising=False
    )
    monkeypatch.setattr(
        service,
        "_call_llm_text",
        lambda system, user: service.web_answer,
        raising=False,
    )
    monkeypatch.setattr(
        app_module.FactCheckService,
        "get_instance",
        classmethod(lambda cls: service),
    )

    with TestClient(app_module.app) as client:
        yield client, service


def test_http_flow_returns_supported_claim_and_public_contract(http_flow):
    client, service = http_flow

    response = client.post(
        "/api/debunk",
        json={"message": "URGENTE: o ministério confirmou a medida em Recife."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "matched",
        "claim": "O ministério confirmou a medida em Recife em 2026",
        "conclusion": "sustentada",
        "evidence_sufficient": True,
        "evidence_level": "forte",
        "explanation": (
            "A fonte resolve a alegação [1].\n\n"
            f"Fonte: checamos.afp.com — Leia mais em: {LOCAL_URL}"
        ),
        "sources": [
            {
                "title": "Checagem local",
                "domain": "checamos.afp.com",
                "url": LOCAL_URL,
                "relevant_excerpt": "O documento oficial confirma a medida.",
                "origin": "local",
                "published_at": "2026-05-11",
            }
        ],
        "probabilistic_assessment": None,
        "trace_id": None,
    }
    assert service.search_queries == [
        "O ministério confirmou a medida em Recife em 2026"
    ]
    assert "score" not in body and "counter_narrative" not in body


def test_http_flow_returns_contradicted_with_external_source(http_flow, monkeypatch):
    client, service = http_flow
    service.hits = []
    monkeypatch.setattr(
        fcs,
        "perform_web_search",
        lambda query, max_results=5: [
            {
                "title": "Nota oficial",
                "snippet": "A nota oficial nega que a medida tenha sido anunciada.",
                "url": WEB_URL,
            }
        ],
    )

    response = client.post(
        "/api/debunk",
        json={"message": "O ministério confirmou a medida em Recife em 2026."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deep_searched"
    assert body["conclusion"] == "contradita"
    assert body["sources"][0]["url"] == WEB_URL
    assert body["sources"][0]["origin"] == "web"


def test_http_flow_abstains_before_retrieval_for_vague_message(http_flow):
    client, service = http_flow
    service.claim_answer = _claim_analysis(None)

    response = client.post(
        "/api/debunk",
        json={"message": "Isso é verdade mesmo?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "abstained"
    assert body["claim"] is None
    assert body["conclusion"] == "inconclusiva"
    assert body["evidence_sufficient"] is False
    assert body["sources"] == []
    assert "qual medida" in body["explanation"]
    assert service.search_queries == []


def test_http_flow_accepts_short_message_and_abstains_when_context_is_missing(http_flow):
    client, service = http_flow
    service.claim_answer = _claim_analysis(None)

    response = client.post("/api/debunk", json={"message": "curta"})

    assert response.status_code == 200
    assert response.json()["conclusion"] == "inconclusiva"
    assert service.search_queries == []
