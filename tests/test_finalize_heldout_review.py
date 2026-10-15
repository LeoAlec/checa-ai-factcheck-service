import pytest

from scripts.finalize_heldout_review import build_error_rows, validate_review


def _case(case_id="gc-001", label="inconclusiva"):
    return {"id": case_id, "gold_conclusion": label, "tags": []}


def test_validate_review_requires_completed_human_decisions():
    with pytest.raises(ValueError, match="decisão humana ausente"):
        validate_review([_case()], [{"id": "gc-001", "gold_conclusion": "inconclusiva", "human_conclusion": ""}])


def test_validate_review_reports_disagreement_without_changing_gold():
    result = validate_review(
        [_case()],
        [{"id": "gc-001", "gold_conclusion": "inconclusiva", "human_conclusion": "contradita"}],
    )
    assert result["agreement_with_gold"] == 0
    assert result["mismatches"] == ["gc-001"]


def test_build_error_rows_only_includes_mistakes():
    dataset = [_case("gc-014", "sustentada"), _case("gc-002", "inconclusiva")]
    evaluation = {
        "cases": [
            {"id": "gc-014", "gold": "sustentada", "pred": "contradita"},
            {"id": "gc-002", "gold": "inconclusiva", "pred": "inconclusiva"},
        ]
    }
    errors = build_error_rows(dataset, evaluation)
    assert [row["id"] for row in errors] == ["gc-014"]
    assert errors[0]["category"] == "referencia_temporal"
