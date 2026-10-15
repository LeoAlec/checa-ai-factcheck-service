"""
src/rag/reranker.py
─────────────────────────────────────────────────────────────────────
Reranker cross-encoder multilíngue opcional (ex.: BAAI/bge-reranker-v2-m3).

Desligado por padrão (`RERANKER_ENABLED=false`): carrega torch/transformers
só quando usado. Em qualquer falha, devolve a ordem original — o reranker
nunca derruba a verificação. O `score` de similaridade é preservado.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

from src.config import RERANKER_ENABLED, RERANKER_MODEL, RERANKER_TOP_N
from src.rag.retriever import Evidence

logger = logging.getLogger("checa-ai.reranker")
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cross_encoder_rerank(
    query: str, evidences: list[Evidence], enabled: bool = RERANKER_ENABLED, top_n: int = RERANKER_TOP_N
) -> list[Evidence]:
    """Repontua os `top_n` melhores candidatos; o restante fica abaixo, na ordem anterior."""
    if not enabled or len(evidences) < 2:
        return evidences
    try:
        ordered = sorted(
            evidences,
            key=lambda e: e.rerank_score if e.rerank_score is not None else e.score,
            reverse=True,
        )
        head, tail = ordered[:top_n], ordered[top_n:]
        raw = _get_model().predict([(query, f"{e.titulo}. {e.texto}") for e in head])
        scored = [replace(e, rerank_score=_sigmoid(float(s))) for e, s in zip(head, raw)]
        floor = min((e.rerank_score for e in scored), default=0.0)
        # Cauda fica sempre abaixo dos reordenados.
        tail = [replace(e, rerank_score=min(e.rerank_score or 0.0, floor) * 0.999) for e in tail]
        return scored + tail
    except Exception:  # noqa: BLE001 - degradação segura
        logger.warning("Reranker cross-encoder indisponível; mantendo a ordem anterior.")
        return evidences
