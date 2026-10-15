"""Human-review export and validation for retrieval false positives."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REVIEW_CATEGORIES = {
    "label_error", "corpus_gap", "entity_mismatch", "time_mismatch", "topic_only",
    "duplicate_or_canonical", "ambiguous_query", "embedding_error", "other",
}

AUDIT_FIELDS = (
    "id", "split", "query", "best_score", "threshold", "rank", "top_url",
    "expected_url", "error_type", "review_category", "review_decision", "review_notes",
)


def build_false_positive_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Mapping[str, Any]],
    threshold: float,
    split: str | None = None,
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for row in rows:
        answered = float(row.get("best_score", 0.0)) >= threshold
        wrong_top1 = row.get("expected_status") == "matched" and row.get("rank") != 1
        negative_answered = row.get("expected_status") == "abstained" and answered
        if not answered or not (wrong_top1 or negative_answered):
            continue
        case = metadata.get(str(row.get("id")), {})
        case_split = case.get("split", row.get("split", ""))
        if split is not None and case_split != split:
            continue
        audit.append({
            "id": row.get("id"),
            "split": case_split,
            "query": case.get("query", ""),
            "best_score": row.get("best_score"),
            "threshold": threshold,
            "rank": row.get("rank"),
            "top_url": row.get("top_url"),
            "expected_url": row.get("expected_url") or case.get("expected_url"),
            "error_type": "wrong_top1" if wrong_top1 else "negative_answered",
            "review_category": "",
            "review_decision": "",
            "review_notes": "",
        })
    return audit


def write_audit_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_review_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        case_id = row.get("id", "<sem id>")
        category = str(row.get("review_category", "")).strip()
        if category not in REVIEW_CATEGORIES:
            raise ValueError(f"categoria inválida no caso {case_id}: {category!r}")
        if not str(row.get("review_decision", "")).strip():
            raise ValueError(f"decisão ausente no caso {case_id}")
        if not str(row.get("review_notes", "")).strip():
            raise ValueError(f"justificativa ausente no caso {case_id}")
