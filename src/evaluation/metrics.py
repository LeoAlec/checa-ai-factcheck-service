"""
src/evaluation/metrics.py
─────────────────────────────────────────────────────────────────────
Métricas puras (sem rede/modelo) para avaliar retrieval, veredito e
abstenção, com intervalos de confiança por bootstrap.

Com amostras pequenas um ponto percentual não significa nada: toda métrica
relevante deve ser reportada com IC (`bootstrap_ci`).
"""

from __future__ import annotations

import math
import random
from typing import Callable, Hashable, Sequence

CONCLUSIONS = ("sustentada", "contradita", "inconclusiva")


# ─── Retrieval ───────────────────────────────────────────────────────
def recall_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """Fração dos documentos relevantes presentes no top-k (0 se não há relevantes)."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[Hashable], relevant: set[Hashable]) -> float:
    for rank, doc in enumerate(ranked, 1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """nDCG@k com relevância binária."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(ranked[:k], 1) if doc in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


# ─── Classificação (conclusão evidencial) ────────────────────────────
def confusion_matrix(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = CONCLUSIONS
) -> dict[str, dict[str, int]]:
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
    return matrix


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = CONCLUSIONS) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else 0.0


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Concordância entre dois anotadores (ex.: gerador × juiz LLM)."""
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


# ─── Abstenção: curva precisão × cobertura ───────────────────────────
def precision_coverage_curve(
    scores: Sequence[float], correct: Sequence[bool], thresholds: Sequence[float]
) -> list[dict]:
    """Para cada limiar, cobertura (fração respondida) e precisão entre as respondidas."""
    total = len(scores)
    curve = []
    for threshold in thresholds:
        answered = [ok for s, ok in zip(scores, correct) if s >= threshold]
        curve.append(
            {
                "threshold": threshold,
                "coverage": len(answered) / total if total else 0.0,
                "precision": sum(answered) / len(answered) if answered else 0.0,
            }
        )
    return curve


def calibrate_threshold(
    scores: Sequence[float],
    should_answer: Sequence[bool],
    candidates: Sequence[float] | None = None,
    min_precision: float = 0.9,
) -> dict:
    """Escolhe o menor limiar com precisão ≥ `min_precision` (maximiza cobertura).

    `should_answer[i]` indica se o melhor documento recuperado era de fato
    relevante para a alegação i. Se nenhum limiar atinge a precisão mínima,
    devolve o de maior F1 e marca `met_target=False`.
    """
    candidates = list(candidates) if candidates else [round(x / 100, 2) for x in range(20, 96, 1)]
    best_f1: dict | None = None
    for threshold in sorted(candidates):
        tp = sum(1 for s, y in zip(scores, should_answer) if s >= threshold and y)
        fp = sum(1 for s, y in zip(scores, should_answer) if s >= threshold and not y)
        fn = sum(1 for s, y in zip(scores, should_answer) if s < threshold and y)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "met_target": precision >= min_precision and tp > 0}
        if row["met_target"]:
            return row
        if best_f1 is None or f1 > best_f1["f1"]:
            best_f1 = row
    return best_f1 or {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0, "met_target": False}


# ─── Bootstrap ───────────────────────────────────────────────────────
def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 13,
) -> dict:
    """IC percentil por bootstrap. Por padrão a estatística é a média."""
    stat = statistic or (lambda xs: sum(xs) / len(xs))
    if not values:
        return {"estimate": None, "low": None, "high": None, "n": 0}
    rng = random.Random(seed)
    data = list(values)
    estimates = sorted(stat([rng.choice(data) for _ in data]) for _ in range(n_resamples))
    low = estimates[int((alpha / 2) * n_resamples)]
    high = estimates[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return {"estimate": stat(data), "low": low, "high": high, "n": len(data)}


def paired_bootstrap_ci(
    gold: Sequence[str],
    pred: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float] = macro_f1,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 13,
) -> dict:
    """IC por bootstrap de pares (gold, pred) para métricas não aditivas (ex.: macro-F1)."""
    if not gold:
        return {"estimate": None, "low": None, "high": None, "n": 0}
    rng = random.Random(seed)
    idx = list(range(len(gold)))
    estimates = []
    for _ in range(n_resamples):
        sample = [rng.choice(idx) for _ in idx]
        estimates.append(metric([gold[i] for i in sample], [pred[i] for i in sample]))
    estimates.sort()
    return {
        "estimate": metric(gold, pred),
        "low": estimates[int((alpha / 2) * n_resamples)],
        "high": estimates[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))],
        "n": len(gold),
    }
