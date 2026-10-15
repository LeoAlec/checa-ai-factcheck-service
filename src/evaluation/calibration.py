"""Pure threshold selection, evaluation and promotion policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

CalibrationStrategy = Literal["max_f1", "min_precision"]
PromotionTier = Literal["none", "pilot", "production"]


@dataclass(frozen=True)
class ThresholdSelection:
    strategy: CalibrationStrategy
    threshold: float
    precision: float
    recall: float
    f1: float
    met_target: bool
    target_precision: float | None


def _should_answer(row: dict[str, Any]) -> bool:
    return row.get("expected_status") == "matched" and row.get("rank") == 1


def evaluate_threshold(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        expected_matched = row.get("expected_status") == "matched"
        correct_match = _should_answer(row)
        answered = float(row["best_score"]) >= threshold
        if expected_matched and answered and correct_match:
            tp += 1
        elif expected_matched:
            fn += 1
            if answered:
                fp += 1
        elif answered:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_cases": len(rows), "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def select_threshold(
    rows: Sequence[dict[str, Any]],
    *,
    strategy: CalibrationStrategy,
    min_precision: float = 0.90,
    candidates: Sequence[float] | None = None,
) -> ThresholdSelection:
    calibration_rows = [row for row in rows if row.get("split") == "calibration"]
    if not calibration_rows:
        raise ValueError("nenhum caso no split calibration")
    thresholds = list(candidates) if candidates is not None else [round(value / 100, 2) for value in range(20, 96)]
    evaluated = [(threshold, evaluate_threshold(calibration_rows, threshold)) for threshold in thresholds]
    if strategy == "min_precision":
        eligible = [item for item in evaluated if item[1]["precision"] >= min_precision and item[1]["tp"] > 0]
        threshold, selected = min(eligible, key=lambda item: item[0]) if eligible else max(
            evaluated, key=lambda item: (item[1]["f1"], item[1]["precision"], item[0])
        )
        return ThresholdSelection(
            strategy=strategy, threshold=threshold,
            precision=float(selected["precision"]), recall=float(selected["recall"]),
            f1=float(selected["f1"]), met_target=bool(eligible),
            target_precision=min_precision,
        )
    if strategy != "max_f1":
        raise ValueError(f"estratégia desconhecida: {strategy}")
    threshold, metrics = max(evaluated, key=lambda item: (item[1]["f1"], item[1]["precision"], item[0]))
    return ThresholdSelection(
        strategy=strategy, threshold=threshold, precision=metrics["precision"],
        recall=metrics["recall"], f1=metrics["f1"], met_target=True,
        target_precision=None,
    )


def build_calibration_artifact(
    *,
    rows: Sequence[dict[str, Any]],
    dataset_path: Path,
    provider: str,
    model: str,
    collection: str,
    strategy: CalibrationStrategy,
    promotion_tier: PromotionTier,
    code_commit: str | None,
    min_precision: float = 0.90,
    minimum_cases: int = 40,
    dataset_label: str | None = None,
) -> dict[str, Any]:
    selection = select_threshold(rows, strategy=strategy, min_precision=min_precision)
    calibration_rows = [row for row in rows if row.get("split") == "calibration"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        raise ValueError("nenhum caso no split test")
    reliable = (
        len(calibration_rows) >= minimum_cases
        and len(test_rows) >= minimum_cases
        and (selection.met_target if strategy == "min_precision" else False)
    )
    if promotion_tier == "production" and not reliable:
        raise ValueError("artefato não confiável não pode ser promovido para produção")
    return {
        "schema_version": 1,
        "provider": provider,
        "model": model,
        "collection": collection,
        "dataset_path": dataset_label or dataset_path.as_posix(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "n_calibration": len(calibration_rows),
        "n_test": len(test_rows),
        "strategy": selection.strategy,
        "target_precision": selection.target_precision,
        "threshold": selection.threshold,
        "calibration_metrics": evaluate_threshold(calibration_rows, selection.threshold),
        "test_metrics": evaluate_threshold(test_rows, selection.threshold),
        "reliable": reliable,
        "promotion_tier": promotion_tier,
        "code_commit": code_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def artifact_is_eligible(artifact: dict[str, Any], *, runtime_tier: str) -> bool:
    promotion_tier = artifact.get("promotion_tier", "none")
    if runtime_tier == "pilot":
        return promotion_tier in {"pilot", "production"}
    if runtime_tier == "production":
        return promotion_tier == "production" and artifact.get("reliable") is True
    return False
