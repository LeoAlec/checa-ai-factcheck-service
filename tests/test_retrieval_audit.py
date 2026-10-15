import csv
from pathlib import Path

import pytest

from src.evaluation.audit import build_false_positive_audit, validate_review_rows, write_audit_csv


def test_exports_negative_matches_and_wrong_top1(tmp_path: Path) -> None:
    rows = [
        {"id": "ok", "expected_status": "matched", "rank": 1, "best_score": 0.8},
        {"id": "wrong", "expected_status": "matched", "rank": 2, "best_score": 0.8},
        {"id": "negative", "expected_status": "abstained", "rank": None, "best_score": 0.7},
        {"id": "safe", "expected_status": "abstained", "rank": None, "best_score": 0.4},
    ]
    metadata = {
        row["id"]: {"query": f"query-{row['id']}", "split": "test", "expected_url": None}
        for row in rows
    }
    audit = build_false_positive_audit(rows, metadata=metadata, threshold=0.65)
    assert [row["id"] for row in audit] == ["wrong", "negative"]
    assert all(row["review_category"] == "" for row in audit)

    output = tmp_path / "audit.csv"
    write_audit_csv(audit, output)
    with output.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_review_validation_rejects_unknown_or_incomplete_decisions() -> None:
    base = {"id": "x", "review_category": "embedding_error", "review_decision": "confirmed", "review_notes": "wrong result"}
    validate_review_rows([base])
    with pytest.raises(ValueError, match="categoria"):
        validate_review_rows([{**base, "review_category": "magic"}])
    with pytest.raises(ValueError, match="decisão"):
        validate_review_rows([{**base, "review_decision": ""}])
