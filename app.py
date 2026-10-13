"""
app.py — Checa-AI FastAPI Application
──────────────────────────────────────────────────────────────────────
Servidor principal. Expõe:
  GET  /            → página home (redireciona para /debunk)
  GET  /debunk      → interface de verificação de boatos
  POST /api/debunk  → conclusão evidencial JSON grounded

O FactCheckService é singleton e inicializado sob demanda na primeira verificação.

Uso:
    ./.venv/bin/uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Garante que src/ está no path quando rodando da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent))
from src import observability
from src.fact_check_service import FactCheckService
from src.models import ClaimRequest, FactCheckResponse
from src.api.v1 import router as v1_router

log = logging.getLogger("checa-ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

BASE_DIR   = Path(__file__).parent
TEMPLATES  = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ─── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # O serviço é inicializado na primeira verificação. Assim, healthchecks e
    # feedback continuam disponíveis mesmo se Qdrant/embeddings estiverem fora.
    log.info("Checa-AI startup — inicialização do pipeline sob demanda.")
    yield
    observability.shutdown()  # envia traces pendentes ao Langfuse
    log.info("🛑 Checa-AI shutdown.")


# ─── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Checa-AI",
    description="Serviço independente de verificação de alegações baseada em evidências",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(v1_router)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ─── Schemas ──────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    value: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)


# ─── Rotas ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/debunk")


@app.get("/debunk", response_class=HTMLResponse, include_in_schema=False)
async def debunk_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request=request,
        name="pages/debunk.html",
        context={"title": "Verificar Boato — Checa-AI"},
    )


@app.post("/api/debunk")
def api_debunk(body: ClaimRequest) -> FactCheckResponse:
    """
    Retorna status operacional e conclusão evidencial separados, além da
    suficiência, nível, explicação e fontes efetivamente usadas.
    """
    try:
        service = FactCheckService.get_instance()
        result  = service.verify_claim(body.message, session_id=body.session_id)
    except RuntimeError as exc:
        # OPENAI_API_KEY ausente ou outro erro de configuração
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("Erro inesperado em /api/debunk: %s", exc)
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")

    return FactCheckResponse.model_validate(result)


@app.post("/api/feedback", status_code=202)
async def api_feedback(body: FeedbackRequest):
    """Registra 👍/👎 do usuário como score `user_feedback` no trace da resposta."""
    recorded = observability.score_trace(
        body.trace_id,
        "user_feedback",
        1.0 if body.value == "up" else 0.0,
        data_type="BOOLEAN",
        comment=body.comment,
        idempotent=True,  # reenvio sobrescreve em vez de duplicar
    )
    if not recorded:
        raise HTTPException(status_code=503, detail="Feedback indisponível no momento.")
    return {"recorded": True}


# ─── Healthcheck ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "checa-ai"}
