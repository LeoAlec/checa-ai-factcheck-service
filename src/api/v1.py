"""Versioned verification contract for backend clients."""

from __future__ import annotations

import hmac
import logging
import os
from typing import Literal

from anyio import CapacityLimiter, to_thread
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import COLLECTION_NAME
from src.fact_check_service import FactCheckService, VerificationGenerationError

log = logging.getLogger(__name__)
router = APIRouter()
_verification_limiter = CapacityLimiter(4)


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain a nonblank character")
        return value


class VerificationSource(BaseModel):
    title: str
    domain: str
    url: str


class VerifyResponse(BaseModel):
    schema_version: Literal[1]
    verdict: Literal["matched", "deep_searched", "insufficient_evidence", "blocked"]
    similarity_score: float | None
    counter_narrative: str | None
    sources: list[VerificationSource]


def require_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    """Defesa em profundidade: com INTERNAL_API_TOKEN definido, exige o header do backend.

    Sem a variável (dev/testes locais) a rota segue aberta; em produção o deploy a exige.
    """
    expected = os.environ.get("INTERNAL_API_TOKEN", "")
    if not expected:
        return
    if not x_internal_token or not hmac.compare_digest(
        x_internal_token.encode(), expected.encode()
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post(
    "/api/v1/verify",
    response_model=VerifyResponse,
    dependencies=[Depends(require_internal_token)],
)
async def verify(body: VerifyRequest) -> VerifyResponse:
    try:
        service = await to_thread.run_sync(FactCheckService.get_instance)
        result = await to_thread.run_sync(
            service.verify_claim_for_backend,
            body.text,
            limiter=_verification_limiter,
        )
        return VerifyResponse.model_validate(result)
    except VerificationGenerationError as exc:
        log.warning("Verification generation failed")
        raise HTTPException(status_code=502, detail="Verification generation failed") from exc
    except Exception as exc:
        log.error("Verification request failed")
        raise HTTPException(status_code=503, detail="Verification unavailable") from exc


@router.get("/ready")
async def ready() -> dict[str, str]:
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI generation key is not configured")
        service = await to_thread.run_sync(FactCheckService.get_instance)
        collection = await to_thread.run_sync(service._qdrant.get_collection, COLLECTION_NAME)
        vectors = collection.config.params.vectors
        size = vectors.get("size") if isinstance(vectors, dict) else vectors.size
        dimension = await to_thread.run_sync(lambda: service._embedder.dimension)
        if size != dimension:
            raise ValueError("Collection vector dimension does not match embedding provider")
    except Exception as exc:
        log.warning("Verification service not ready")
        raise HTTPException(status_code=503, detail="Verification index unavailable") from exc
    return {"status": "ready"}
