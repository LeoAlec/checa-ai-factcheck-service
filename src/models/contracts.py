"""Public API and LLM contracts for evidence-based claim verification."""

from __future__ import annotations

from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OperationalStatus(StrEnum):
    MATCHED = "matched"
    DEEP_SEARCHED = "deep_searched"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"


class EvidenceConclusion(StrEnum):
    SUPPORTED = "sustentada"
    CONTRADICTED = "contradita"
    INCONCLUSIVE = "inconclusiva"


class EvidenceLevel(StrEnum):
    STRONG = "forte"
    MODERATE = "moderada"
    LIMITED = "limitada"
    NONE = "ausente"


class SourceOrigin(StrEnum):
    LOCAL = "local"
    WEB = "web"


def validate_http_url(value: str) -> str:
    value = str(value).strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("a URL deve ser HTTP(S) e possuir domínio")
    if parsed.username or parsed.password:
        raise ValueError("credenciais embutidas na URL não são permitidas")
    return value


class ClaimRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2_000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("message")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("a mensagem contém caracteres de controle inválidos")
        return value


class EvidenceSource(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=500)
    domain: str = Field(min_length=1, max_length=253)
    url: str
    relevant_excerpt: str = Field(min_length=1, max_length=2_000)
    origin: SourceOrigin
    published_at: str | None = Field(default=None, max_length=64)

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, value: str) -> str:
        return validate_http_url(value)

    @model_validator(mode="after")
    def normalize_domain(self) -> "EvidenceSource":
        hostname = (urlsplit(self.url).hostname or "").lower()
        supplied = self.domain.lower().removeprefix("www.")
        if supplied and hostname.removeprefix("www.") != supplied:
            self.domain = hostname.removeprefix("www.")
        return self


class ProbabilisticAssessment(BaseModel):
    """Reserved extension point; no classifier is implemented in phase 1."""

    model_name: str
    label: str
    confidence: float = Field(ge=0, le=1)


class ClaimAnalysis(BaseModel):
    """Structured assessment of whether an input contains a checkable claim."""

    has_verifiable_claim: bool
    claim: str | None = Field(default=None, min_length=3, max_length=1_000)
    missing_context: list[str] = Field(default_factory=list, max_length=5)
    explanation: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def enforce_claim_consistency(self) -> "ClaimAnalysis":
        self.missing_context = list(dict.fromkeys(self.missing_context))
        if not self.has_verifiable_claim or self.missing_context:
            self.has_verifiable_claim = False
            self.claim = None
        elif not self.claim:
            self.has_verifiable_claim = False
        return self


class FactCheckResponse(BaseModel):
    status: OperationalStatus
    claim: str | None = None
    conclusion: EvidenceConclusion
    evidence_sufficient: bool
    evidence_level: EvidenceLevel
    explanation: str = Field(min_length=1)
    sources: list[EvidenceSource] = Field(default_factory=list)
    probabilistic_assessment: ProbabilisticAssessment | None = None
    trace_id: str | None = None

    @model_validator(mode="after")
    def enforce_inconclusive_contract(self) -> "FactCheckResponse":
        if self.conclusion is EvidenceConclusion.INCONCLUSIVE:
            self.evidence_sufficient = False
        if not self.sources:
            self.evidence_sufficient = False
            if self.evidence_level is not EvidenceLevel.NONE:
                self.evidence_level = EvidenceLevel.NONE
        return self


class LLMDecision(BaseModel):
    """Strict grounded output expected from the language model."""

    conclusion: EvidenceConclusion
    evidence_sufficient: bool
    evidence_level: EvidenceLevel
    explanation: str = Field(min_length=1)
    source_urls: list[str] = Field(default_factory=list)

    @field_validator("source_urls")
    @classmethod
    def validate_source_urls(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(validate_http_url(value) for value in values))

    @model_validator(mode="after")
    def inconclusive_is_never_sufficient(self) -> "LLMDecision":
        if self.conclusion is EvidenceConclusion.INCONCLUSIVE:
            self.evidence_sufficient = False
        return self
