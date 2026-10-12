"""Deterministic aggregation for adjudicated human reviews.

This module deliberately evaluates the observable evidence flow rather than a
classifier. Reviewers inspect the extracted claim, returned sources, conclusion,
abstention and cited URLs in a captured service response.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "claim_extraction",
    "source_relevance",
    "conclusion_correct",
    "abstention_correct",
    "citation_grounding",
)
SCORES = {"pass", "fail", "na"}
THRESHOLDS = {
    "claim_extraction": 0.90,
    "source_relevance": 0.90,
    "conclusion_correct": 0.90,
    "abstention_correct": 0.95,
    "citation_grounding": 1.00,
}
MIN_CASES = 20
MIN_REVIEWERS = 2


@dataclass(frozen=True)
class HumanReview:
    case_id: str
    reviewer_id: str
    review_status: str
    scores: dict[str, str]
    notes: str


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    """Load captured run cases from JSONL, rejecting duplicate or malformed IDs."""
    cases: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"caso sem case_id na linha {line_number}")
        if case_id in cases:
            raise ValueError(f"case_id duplicado: {case_id}")
        if not str(row.get("input", "")).strip() or "response" not in row:
            raise ValueError(f"caso {case_id} precisa de input e response")
        cases[case_id] = row
    if not cases:
        raise ValueError("nenhum caso encontrado")
    return cases


def load_reviews(path: Path, cases: dict[str, dict[str, Any]]) -> list[HumanReview]:
    """Load final human adjudications, validating every score and case reference."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "reviewer_id", "review_status", *DIMENSIONS, "notes"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV deve conter: {', '.join(sorted(required))}")
        reviews: list[HumanReview] = []
        for line_number, row in enumerate(reader, 2):
            case_id = (row.get("case_id") or "").strip()
            reviewer_id = (row.get("reviewer_id") or "").strip()
            if case_id not in cases:
                raise ValueError(f"linha {line_number}: case_id desconhecido: {case_id}")
            if not reviewer_id:
                raise ValueError(f"linha {line_number}: reviewer_id é obrigatório")
            scores = {dimension: (row.get(dimension) or "").strip().lower() for dimension in DIMENSIONS}
            invalid = {key: value for key, value in scores.items() if value not in SCORES}
            if invalid:
                raise ValueError(f"linha {line_number}: score inválido: {invalid}")
            reviews.append(
                HumanReview(
                    case_id=case_id,
                    reviewer_id=reviewer_id,
                    review_status=(row.get("review_status") or "").strip().lower(),
                    scores=scores,
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return reviews


def build_quality_report(
    cases: dict[str, dict[str, Any]], reviews: list[HumanReview]
) -> dict[str, Any]:
    """Aggregate only approved adjudications and evaluate the explicit release gate."""
    approved = [review for review in reviews if review.review_status == "approved"]
    by_case: dict[str, list[HumanReview]] = {}
    for review in approved:
        by_case.setdefault(review.case_id, []).append(review)

    duplicate_cases = sorted(case_id for case_id, rows in by_case.items() if len(rows) > 1)
    missing_cases = sorted(set(cases) - set(by_case))
    reviewers = sorted({review.reviewer_id for review in approved})
    metrics: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        values = [review.scores[dimension] for review in approved if review.scores[dimension] != "na"]
        passed = values.count("pass")
        rate = passed / len(values) if values else None
        threshold = THRESHOLDS[dimension]
        metrics[dimension] = {
            "passed": passed,
            "applicable": len(values),
            "rate": rate,
            "threshold": threshold,
            "meets_threshold": rate is not None and rate >= threshold,
        }

    blockers: list[str] = []
    if len(cases) < MIN_CASES:
        blockers.append(f"amostra insuficiente: {len(cases)}/{MIN_CASES} casos")
    if len(reviewers) < MIN_REVIEWERS:
        blockers.append(f"revisores insuficientes: {len(reviewers)}/{MIN_REVIEWERS}")
    if missing_cases:
        blockers.append(f"casos sem adjudicação aprovada: {', '.join(missing_cases)}")
    if duplicate_cases:
        blockers.append(f"mais de uma adjudicação aprovada: {', '.join(duplicate_cases)}")
    for dimension, metric in metrics.items():
        if not metric["meets_threshold"]:
            blockers.append(
                f"{dimension}: {metric['passed']}/{metric['applicable']} abaixo de {metric['threshold']:.0%}"
            )

    return {
        "decision": "approved_for_backend_integration" if not blockers else "not_ready",
        "cases": len(cases),
        "approved_reviews": len(approved),
        "reviewers": reviewers,
        "metrics": metrics,
        "blockers": blockers,
        "criteria": {
            "minimum_cases": MIN_CASES,
            "minimum_reviewers": MIN_REVIEWERS,
            "thresholds": THRESHOLDS,
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    """Render the machine report into a reviewable Markdown decision record."""
    lines = [
        "# Avaliação humana do fluxo evidencial",
        "",
        f"**Decisão:** `{report['decision']}`",
        "",
        f"Casos: {report['cases']} · Adjudicações aprovadas: {report['approved_reviews']} · "
        f"Revisores: {len(report['reviewers'])}",
        "",
        "| Dimensão | Resultado | Critério |",
        "| --- | --- | --- |",
    ]
    for dimension, metric in report["metrics"].items():
        rate = "N/A" if metric["rate"] is None else f"{metric['rate']:.1%}"
        lines.append(
            f"| {dimension} | {metric['passed']}/{metric['applicable']} ({rate}) | "
            f">= {metric['threshold']:.0%} |"
        )
    lines.extend(["", "## Bloqueios"])
    lines.extend(f"- {blocker}" for blocker in report["blockers"] or ["Nenhum."])
    return "\n".join(lines) + "\n"
