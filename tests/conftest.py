import os
import json

# Testes comuns nunca enviam traces ao Langfuse real. O smoke explicitamente
# habilitado preserva a configuração do ambiente para validar tracing opcional.
if not os.environ.get("RUN_E2E"):
    os.environ["LANGFUSE_ENABLED"] = "false"

import pytest


@pytest.fixture(autouse=True)
def offline_safety_and_web(monkeypatch, request):
    """Moderação (OpenAI) e DeepSearch (Tavily) são rede: nos testes ficam inertes."""
    if request.node.get_closest_marker("live_e2e"):
        return

    import src.fact_check_service as fcs
    from src.safety.moderation import ModerationResult

    monkeypatch.setattr(fcs, "check_moderation", lambda text: ModerationResult(is_safe=True))
    monkeypatch.setattr(fcs, "validate_output", lambda text: ModerationResult(is_safe=True))
    monkeypatch.setattr(fcs, "perform_web_search", lambda query, max_results=5: [])
    def analyze_claim(_self, _system, user):
        message = user.split("--- ENTRADA ---\n", 1)[-1].split(
            "\n\n--- CONTEXTO DA URL", 1
        )[0].strip()
        return json.dumps(
            {
                "has_verifiable_claim": True,
                "claim": message,
                "missing_context": [],
                "explanation": "Alegação verificável para o teste.",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(fcs.FactCheckService, "_call_claim_analyzer", analyze_claim)
    # Síntese do DeepSearch nunca chama a OpenAI nos testes, mas preserva o
    # contrato estrito para não mascarar falhas de parsing.
    monkeypatch.setattr(
        fcs.FactCheckService,
        "_call_llm_text",
        lambda self, s, u: json.dumps(
            {
                "conclusion": "contradita",
                "evidence_sufficient": True,
                "evidence_level": "forte",
                "explanation": "A evidência externa resolve a alegação [1].",
                "source_urls": [],
            },
            ensure_ascii=False,
        ),
    )
