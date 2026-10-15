from scripts.build_local_llm_benchmark import build_rows


def test_build_rows_preserves_evidence_and_gold_without_extra_fields():
    rows = build_rows([{
        "id": "gc-001", "claim": "Alegação", "gold_conclusion": "inconclusiva", "tags": ["entity"],
        "evidence": {"titulo": "Laudo", "texto": "Texto"},
    }])
    assert rows == [{
        "id": "gc-001", "claim": "Alegação", "evidence_title": "Laudo", "evidence_text": "Texto",
        "gold_conclusion": "inconclusiva", "tags": ["entity"],
    }]
