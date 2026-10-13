"""Classification and probability-quality metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-probabilities, kind="stable")
    ranked = labels[order]
    precision_at_rank = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision_at_rank * ranked) / positives)


def reliability_curve(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10
) -> list[dict[str, float | int]]:
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(y_prob, edges[1:-1], right=False), bins - 1)
    curve: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = assignments == index
        if not np.any(mask):
            continue
        curve.append(
            {
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_probability": float(y_prob[mask].mean()),
                "fraction_positive": float(y_true[mask].mean()),
            }
        )
    return curve


def expected_calibration_error(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10
) -> float:
    total = len(labels)
    if not total:
        raise ValueError("ECE requer pelo menos uma observação")
    return float(
        sum(
            point["count"] / total
            * abs(point["fraction_positive"] - point["mean_probability"])
            for point in reliability_curve(labels, probabilities, bins=bins)
        )
    )


def classification_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *, threshold: float = 0.5
) -> dict[str, object]:
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(y_prob) or not len(y_true):
        raise ValueError("rótulos e probabilidades devem ter o mesmo tamanho não vazio")
    predicted = (y_prob >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (predicted == 0)))
    fp = int(np.sum((y_true == 0) & (predicted == 1)))
    fn = int(np.sum((y_true == 1) & (predicted == 0)))
    tp = int(np.sum((y_true == 1) & (predicted == 1)))
    has_both_classes = set(np.unique(y_true)) == {0, 1}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "accuracy": float(np.mean(y_true == predicted)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": _roc_auc(y_true, y_prob) if has_both_classes else None,
        "pr_auc": _average_precision(y_true, y_prob) if has_both_classes else None,
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
        "ece": expected_calibration_error(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "reliability_curve": reliability_curve(y_true, y_prob),
    }
