"""
src/rag/nli.py
─────────────────────────────────────────────────────────────────────
Verificador NLI auxiliar (premissa = trecho do laudo, hipótese = alegação).

NÃO decide o veredito: é uma segunda opinião barata. A discordância entre o
LLM e o NLI é registrada (observabilidade) e pode forçar a abstenção quando
`NLI_ABSTAIN_ON_DISAGREEMENT=true`. Desligado por padrão; carrega o modelo
sob demanda e falha de forma segura.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import NLI_ENABLED, NLI_MODEL

logger = logging.getLogger("checa-ai.nli")
_pipe = None

# Rótulos NLI → conclusão evidencial. Um laudo que "contradiz" a alegação a refuta.
_TO_CONCLUSION = {"entailment": "sustentada", "contradiction": "contradita", "neutral": "inconclusiva"}


@dataclass(frozen=True)
class NLIResult:
    label: str  # entailment | contradiction | neutral
    scores: dict[str, float]

    @property
    def conclusion(self) -> str:
        return _TO_CONCLUSION[self.label]


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline

        _pipe = pipeline("text-classification", model=NLI_MODEL, top_k=None)
    return _pipe


def _normalize_label(label: str) -> str:
    label = label.lower()
    for key in ("entail", "contradict", "neutral"):
        if key in label:
            return {"entail": "entailment", "contradict": "contradiction", "neutral": "neutral"}[key]
    return "neutral"


def classify(claim: str, premise: str) -> NLIResult | None:
    """Rótulo NLI do par (premissa, alegação) ou `None` se desligado/indisponível."""
    if not NLI_ENABLED or not claim.strip() or not premise.strip():
        return None
    try:
        out = _get_pipe()({"text": premise[:1500], "text_pair": claim})
        rows = out[0] if out and isinstance(out[0], list) else out
        scores = {_normalize_label(r["label"]): float(r["score"]) for r in rows}
        return NLIResult(max(scores, key=scores.get), scores)
    except Exception:  # noqa: BLE001
        logger.warning("NLI indisponível; ignorando verificação auxiliar.")
        return None


def agrees(llm_conclusion: str, result: NLIResult | None) -> bool | None:
    """True/False se o NLI concorda com o LLM; None quando não há opinião do NLI."""
    return None if result is None else result.conclusion == llm_conclusion
