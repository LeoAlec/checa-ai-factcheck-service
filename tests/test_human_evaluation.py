import csv
import json

import pytest

from src.human_evaluation.quality_gate import (
    DIMENSIONS,
    build_quality_report,
    load_cases,
    load_reviews,
)


def _write_cases(path, count=20):
    path.write_text(
        "\n".join(
            json.dumps({"case_id": f"case-{index}", "input": "texto", "response": {}})
            for index in range(count)
        )
        + "\n",
        encoding="utf-8",
    )


def _write_reviews(path, case_ids, *, fail_dimension=None):
    fields = ["case_id", "reviewer_id", "review_status", *DIMENSIONS, "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, case_id in enumerate(case_ids):
            writer.writerow(
                {
                    "case_id": case_id,
                    "reviewer_id": f"reviewer-{index % 2}",
                    "review_status": "approved",
                    **{dimension: "fail" if dimension == fail_dimension and index == 0 else "pass" for dimension in DIMENSIONS},
                    "notes": "ok",
                }
            )


def test_quality_gate_approves_complete_human_review(tmp_path):
    cases_path = tmp_path / "responses.jsonl"
    reviews_path = tmp_path / "reviews.csv"
    _write_cases(cases_path)
    cases = load_cases(cases_path)
    _write_reviews(reviews_path, cases)

    report = build_quality_report(cases, load_reviews(reviews_path, cases))

    assert report["decision"] == "approved_for_backend_integration"
    assert report["metrics"]["citation_grounding"]["rate"] == 1.0


def test_quality_gate_blocks_insufficient_or_failed_human_review(tmp_path):
    cases_path = tmp_path / "responses.jsonl"
    reviews_path = tmp_path / "reviews.csv"
    _write_cases(cases_path, count=2)
    cases = load_cases(cases_path)
    _write_reviews(reviews_path, cases, fail_dimension="citation_grounding")

    report = build_quality_report(cases, load_reviews(reviews_path, cases))

    assert report["decision"] == "not_ready"
    assert any("amostra insuficiente" in blocker for blocker in report["blockers"])
    assert any("citation_grounding" in blocker for blocker in report["blockers"])


def test_human_review_rejects_unknown_scores(tmp_path):
    cases_path = tmp_path / "responses.jsonl"
    reviews_path = tmp_path / "reviews.csv"
    _write_cases(cases_path, count=1)
    cases = load_cases(cases_path)
    _write_reviews(reviews_path, cases)
    reviews_path.write_text(reviews_path.read_text(encoding="utf-8").replace("pass", "maybe", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="score inválido"):
        load_reviews(reviews_path, cases)
