"""
src/rag/guardrails.py
─────────────────────────────────────────────────────────────────────
Guardrails determinísticos (sem custo extra de LLM) em volta da geração:

  1. Saída estruturada: o LLM responde {texto, fontes_usadas}, validado por
     Pydantic. Resposta fora do schema não derruba o pipeline: o texto cru
     é aproveitado e as URLs são extraídas por regex.
  2. Grounding de links: toda URL citada (campo ou texto) precisa estar entre
     as evidências recuperadas; link inventado é removido da resposta.
  3. Prompt injection nos laudos raspados: chunks de risco alto são excluídos
     do contexto; os de risco médio entram, mas ficam registrados.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import PrivateAttr, ValidationError

from src.models import ClaimAnalysis, EvidenceConclusion, EvidenceLevel, LLMDecision
from src.rag.preprocessing import assess_prompt_injection

_URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_TRAILING_PUNCT = ".,;:!?"
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


class GeneratedAnswer(LLMDecision):
    """Contrato do LLM com propriedades legadas para consumidores antigos."""

    _legacy_contract: bool = PrivateAttr(default=False)

    @property
    def texto(self) -> str:
        return self.explanation

    @property
    def fontes_usadas(self) -> list[str]:
        return self.source_urls

    @property
    def laudos_tratam_alegacao(self) -> bool:
        return self.evidence_sufficient


# JSON Schema estrito (OpenAI structured outputs exige additionalProperties=false
# e todos os campos em `required`).
RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "conclusao_evidencial",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "conclusion": {
                    "type": "string",
                    "enum": ["sustentada", "contradita", "inconclusiva"],
                },
                "evidence_sufficient": {"type": "boolean"},
                "evidence_level": {
                    "type": "string",
                    "enum": ["forte", "moderada", "limitada", "ausente"],
                },
                "explanation": {"type": "string"},
                "source_urls": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "conclusion",
                "evidence_sufficient",
                "evidence_level",
                "explanation",
                "source_urls",
            ],
            "additionalProperties": False,
        },
    },
}

CLAIM_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "analise_da_alegacao",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "has_verifiable_claim": {"type": "boolean"},
                "claim": {"type": ["string", "null"]},
                "missing_context": {"type": "array", "items": {"type": "string"}},
                "explanation": {"type": "string"},
            },
            "required": [
                "has_verifiable_claim",
                "claim",
                "missing_context",
                "explanation",
            ],
            "additionalProperties": False,
        },
    },
}


def parse_claim_analysis(raw: str) -> ClaimAnalysis:
    """Fail closed when the analyzer does not honor its strict contract."""
    try:
        return ClaimAnalysis.model_validate_json((raw or "").strip())
    except (ValueError, ValidationError):
        return ClaimAnalysis(
            has_verifiable_claim=False,
            claim=None,
            missing_context=["uma alegacao factual clara"],
            explanation=(
                "Nao foi possivel identificar com seguranca uma alegacao verificavel "
                "na mensagem recebida."
            ),
        )


def normalize_url(url: str) -> str:
    return url.strip().rstrip(_TRAILING_PUNCT)


def parse_answer(raw: str) -> tuple[GeneratedAnswer, bool]:
    """Valida a saída do LLM. Retorna (resposta, estruturada?)."""
    raw = (raw or "").strip()
    try:
        payload = json.loads(raw)
        legacy_contract = isinstance(payload, dict) and "texto" in payload
        if legacy_contract:
            payload = {
                # O contrato legado não traz uma conclusão evidencial explícita.
                # Nunca inferimos ``contradita`` só porque ele afirma cobertura.
                "conclusion": "inconclusiva",
                "evidence_sufficient": False,
                "evidence_level": "ausente",
                "explanation": payload.get("texto") or "-",
                "source_urls": payload.get("fontes_usadas") or [],
            }
        answer = GeneratedAnswer.model_validate(payload)
        answer._legacy_contract = legacy_contract
        return answer, True
    except (ValueError, ValidationError):
        # A resposta não passou pelo contrato estrito. Não é seguro inferir uma
        # conclusão a partir do texto cru, nem considerar URLs nele como fontes
        # efetivamente usadas: uma saída inválida não pode contradizer a alegação.
        return GeneratedAnswer(
            conclusion=EvidenceConclusion.INCONCLUSIVE,
            evidence_sufficient=False,
            evidence_level=EvidenceLevel.NONE,
            explanation=(
                "A síntese retornou um formato inválido; faltam evidências "
                "estruturadas para concluir a verificação."
            ),
            source_urls=[],
        ), False


@dataclass(frozen=True)
class GroundingReport:
    structured: bool
    ungrounded_urls: tuple[str, ...] = ()
    invalid_citations: tuple[int, ...] = ()
    sanitized: bool = False
    used_urls: tuple[str, ...] = field(default=())

    @property
    def grounded(self) -> bool:
        return not self.ungrounded_urls

    def to_dict(self) -> dict:
        return {
            "structured": self.structured,
            "ungrounded_urls": list(self.ungrounded_urls),
            "invalid_citations": list(self.invalid_citations),
            "sanitized": self.sanitized,
        }


def enforce_grounding(
    answer: GeneratedAnswer,
    allowed: list[tuple[str, str]],
    structured: bool = True,
    *,
    append_credit: bool = True,
) -> tuple[str, GroundingReport]:
    """
    `allowed`: [(url, dominio)] das evidências recuperadas.
    Remove da resposta qualquer URL fora do contexto e garante que ao menos
    uma fonte real seja creditada. Retorna (texto_final, relatório).
    """
    allowed_ordered = [(normalize_url(url), domain) for url, domain in allowed]
    allowed_urls = {url for url, _ in allowed_ordered}
    cited = {normalize_url(u) for u in answer.fontes_usadas} | {
        normalize_url(u) for u in _URL_RE.findall(answer.texto)
    }
    bad = sorted(cited - allowed_urls)
    texto = answer.texto.strip()
    citation_numbers = {int(value) for value in _CITATION_RE.findall(texto)}
    invalid_citations = sorted(
        number for number in citation_numbers if number < 1 or number > len(allowed_ordered)
    )

    if bad:
        kept: list[str] = []
        for line in texto.splitlines():
            urls = {normalize_url(u) for u in _URL_RE.findall(line)}
            if urls & set(bad):
                if line.lstrip().lower().startswith("fonte"):
                    continue  # linha de crédito com link inventado: descarta
                for u in urls & set(bad):
                    line = line.replace(u, "")
            kept.append(line.rstrip())
        texto = "\n".join(kept).strip()

    for number in invalid_citations:
        texto = texto.replace(f"[{number}]", "")

    present = {normalize_url(u) for u in _URL_RE.findall(texto)} & allowed_urls
    declared = {normalize_url(url) for url in answer.fontes_usadas} & allowed_urls
    indexed = {
        allowed_ordered[number - 1][0]
        for number in citation_numbers
        if 1 <= number <= len(allowed_ordered)
    }
    used = present | declared | indexed
    if (
        not used
        and allowed
        and answer.conclusion is not EvidenceConclusion.INCONCLUSIVE
        and (bad or not structured or answer._legacy_contract)
    ):
        used = allowed_urls
    if used and not present and append_credit:
        credit = "\n".join(
            f"Fonte: {domain} — Leia mais em: {url}"
            for url, domain in allowed_ordered
            if url in used
        )
        texto = f"{texto}\n\n{credit}".strip()
        present = used

    return texto, GroundingReport(
        structured=structured,
        ungrounded_urls=tuple(bad),
        invalid_citations=tuple(invalid_citations),
        sanitized=bool(bad or invalid_citations),
        used_urls=tuple(sorted(used)),
    )


def injection_risk(text: str, payload_risk: str = "") -> str:
    """Maior risco entre o gravado na ingestão e o recalculado agora (cobre payloads antigos)."""
    now = assess_prompt_injection(text).risk
    return max((now, payload_risk or "low"), key=lambda r: _RISK_ORDER.get(r, 0))
