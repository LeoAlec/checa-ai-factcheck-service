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

from src.config import NLI_DEVICE, NLI_ENABLED, NLI_MODEL, NLI_MODEL_REVISION

logger = logging.getLogger("checa-ai.nli")
_pipe = None
_unavailable = False

# Rótulos NLI → conclusão evidencial. Um laudo que "contradiz" a alegação a refuta.
_TO_CONCLUSION = {"entailment": "sustentada", "contradiction": "contradita", "neutral": "inconclusiva"}


@dataclass(frozen=True)
class NLIResult:
    label: str  # entailment | contradiction | neutral
    scores: dict[str, float]

    @property
    def conclusion(self) -> str:
        return _TO_CONCLUSION[self.label]


def _resolve_device(torch_module, configured: str) -> int | str:
    """Traduz configuração estável em dispositivo aceito pelo pipeline."""
    if configured == "auto":
        if torch_module.cuda.is_available():
            return 0
        mps = getattr(torch_module.backends, "mps", None)
        return "mps" if mps and mps.is_available() else -1
    if configured == "cuda":
        return 0
    if configured == "mps":
        return "mps"
    if configured == "cpu":
        return -1
    raise ValueError(f"NLI_DEVICE inválido: {configured!r}")


def _get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline

        device = _resolve_device(torch, NLI_DEVICE)
        logger.info("Carregando NLI model=%s revision=%s device=%s", NLI_MODEL, NLI_MODEL_REVISION, device)
        _pipe = pipeline(
            "text-classification",
            model=NLI_MODEL,
            revision=NLI_MODEL_REVISION,
            top_k=None,
            device=device,
            truncation=True,
            max_length=512,
        )
    return _pipe


def _normalize_label(label: str) -> str:
    label = label.lower()
    for key in ("entail", "contradict", "neutral"):
        if key in label:
            return {"entail": "entailment", "contradict": "contradiction", "neutral": "neutral"}[key]
    return "neutral"


def classify(claim: str, premise: str) -> NLIResult | None:
    """Rótulo NLI do par (premissa, alegação) ou `None` se desligado/indisponível."""
    global _unavailable
    if not NLI_ENABLED or _unavailable or not claim.strip() or not premise.strip():
        return None
    try:
        out = _get_pipe()({"text": premise[:1500], "text_pair": claim})
        rows = out[0] if out and isinstance(out[0], list) else out
        scores = {_normalize_label(r["label"]): float(r["score"]) for r in rows}
        return NLIResult(max(scores, key=scores.get), scores)
    except Exception:  # noqa: BLE001
        _unavailable = True
        logger.warning("NLI indisponível; ignorando verificação auxiliar.")
        return None


def agrees(llm_conclusion: str, result: NLIResult | None) -> bool | None:
    """True/False se o NLI concorda com o LLM; None quando não há opinião do NLI."""
    return None if result is None else result.conclusion == llm_conclusion
