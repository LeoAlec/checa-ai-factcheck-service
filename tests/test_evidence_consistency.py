import logging

from src.fact_check_service import FactCheckService
from src.models import EvidenceConclusion, EvidenceLevel
from src.rag.guardrails import GeneratedAnswer, enforce_evidence_consistency


def test_factual_conclusion_without_used_evidence_fails_closed() -> None:
    answer = GeneratedAnswer(
        conclusion=EvidenceConclusion.CONTRADICTED,
        evidence_sufficient=True,
        evidence_level=EvidenceLevel.STRONG,
        explanation="A alegação é contradita.",
        source_urls=["https://example.com/check"],
    )
    changed = enforce_evidence_consistency(answer, used_urls=())
    assert changed
    assert answer.conclusion is EvidenceConclusion.INCONCLUSIVE
    assert not answer.evidence_sufficient
    assert answer.evidence_level is EvidenceLevel.NONE


def test_output_block_log_contains_stage_but_not_content(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="checa-ai.fact-check"):
        FactCheckService._log_output_block("local_synthesis", "unsafe", trace_id="trace-1")
    assert "stage=local_synthesis" in caplog.text
    assert "category=unsafe" in caplog.text
    assert "trace_id=trace-1" in caplog.text
