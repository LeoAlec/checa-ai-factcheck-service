"""Evaluation orchestration, deliberately disconnected from the production API."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .adapters import ClassifierAdapter
from .artifacts import ArtifactInfo
from .calibration import SigmoidCalibrator, TemperatureCalibrator
from .contracts import DatasetRecord, DatasetSplit
from .metrics import classification_metrics


def _predict_timed(adapter: ClassifierAdapter, records: list[DatasetRecord]) -> tuple[np.ndarray, float]:
    started = perf_counter()
    probabilities = adapter.predict_probability_false([record.text for record in records])
    elapsed_ms = (perf_counter() - started) * 1_000
    if len(probabilities) != len(records):
        raise ValueError("modelo retornou quantidade de previsões diferente da entrada")
    return probabilities, elapsed_ms


def _errors(
    records: list[DatasetRecord], probabilities: np.ndarray, threshold: float
) -> dict[str, list[dict[str, object]]]:
    predictions = probabilities >= threshold
    result: dict[str, list[dict[str, object]]] = {"false_positives": [], "false_negatives": []}
    for record, probability, prediction in zip(records, probabilities, predictions, strict=True):
        item = {
            "id": record.id,
            "text": record.text,
            "topic": record.topic,
            "source": record.source,
            "probability_false": float(probability),
        }
        if prediction and record.binary_label == 0:
            result["false_positives"].append(item)
        elif not prediction and record.binary_label == 1:
            result["false_negatives"].append(item)
    return result


def evaluate_model(
    adapter: ClassifierAdapter,
    artifact: ArtifactInfo,
    records: list[DatasetRecord],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    calibration_records = [record for record in records if record.split is DatasetSplit.CALIBRATION]
    test_records = [record for record in records if record.split is DatasetSplit.TEST]
    calibration_probabilities, calibration_latency = _predict_timed(adapter, calibration_records)
    test_probabilities, test_latency = _predict_timed(adapter, test_records)
    calibration_labels = np.array([record.binary_label for record in calibration_records])
    test_labels = np.array([record.binary_label for record in test_records])

    calibrator = SigmoidCalibrator() if adapter.model_name == "tfidf" else TemperatureCalibrator()
    calibrator.fit(calibration_probabilities, calibration_labels)
    calibrated_test = calibrator.transform(test_probabilities)

    topics: dict[str, object] = {}
    for topic in sorted({record.topic for record in test_records}):
        indexes = [index for index, record in enumerate(test_records) if record.topic == topic]
        topics[topic] = classification_metrics(
            test_labels[indexes], calibrated_test[indexes], threshold=threshold
        )

    return {
        "status": "evaluated",
        "model_name": adapter.model_name,
        "model_version": adapter.model_version,
        "artifact": artifact.as_dict(),
        "dataset": {
            "calibration_records": len(calibration_records),
            "test_records": len(test_records),
            "training_records_used": 0,
        },
        "calibration": {
            "calibrated": True,
            "method": calibrator.method,
            "fit_split": "calibration",
            "evaluation_split": "test",
            "parameters": calibrator.parameters(),
            "limitations": [
                "a calibração só é transferível se o conjunto externo representar o uso real",
                "o conjunto test não foi usado no ajuste do calibrador",
            ],
        },
        "metrics": {
            "raw": classification_metrics(test_labels, test_probabilities, threshold=threshold),
            "calibrated": classification_metrics(test_labels, calibrated_test, threshold=threshold),
            "by_topic": topics,
        },
        "errors": _errors(test_records, calibrated_test, threshold),
        "latency": {
            "calibration_total_ms": calibration_latency,
            "test_total_ms": test_latency,
            "test_ms_per_message": test_latency / len(test_records),
            "device": getattr(adapter, "device", "cpu"),
            "batch_size": getattr(adapter, "batch_size", None),
        },
        "cost": {
            "artifact_size_bytes": artifact.size_bytes,
            "monetary_inference_cost": None,
            "note": "execução local; comparar tamanho e latência medidos no mesmo hardware",
        },
        "limitations": adapter.limitations,
    }


def unavailable_result(name: str, artifact: ArtifactInfo, reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "model_name": name,
        "artifact": artifact.as_dict(),
        "reason": reason,
        "metrics": None,
    }
