"""Train and evaluate transparent baselines for WhatsApp textual risk."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def _expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (probabilities < upper if index < bins - 1 else probabilities <= upper)
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return result


def evaluate(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision_misinformation": precision_score(labels, predictions, zero_division=0),
        "recall_misinformation": recall_score(labels, predictions, zero_division=0),
        "f1_misinformation": f1_score(labels, predictions, zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "brier": brier_score_loss(labels, probabilities),
        "ece": _expected_calibration_error(labels, probabilities),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select on validation data only, maximizing macro-F1."""
    candidates = np.linspace(0.05, 0.95, 181)
    scores = np.asarray(
        [f1_score(labels, probabilities >= threshold, average="macro", zero_division=0) for threshold in candidates]
    )
    best = np.flatnonzero(scores == scores.max())
    index = best[np.argmin(np.abs(candidates[best] - 0.5))]
    return float(candidates[index])


def _models(seed: int) -> dict[str, Pipeline]:
    classifier = lambda: LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)
    return {
        "word_tfidf_logreg": Pipeline(
            [("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)), ("classifier", classifier())]
        ),
        "char_tfidf_logreg": Pipeline(
            [("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=150_000, sublinear_tf=True)), ("classifier", classifier())]
        ),
    }


def train_baselines(dataset_path: str | Path, output_dir: str | Path, *, seed: int = 42) -> dict[str, object]:
    frame = pd.read_csv(dataset_path)
    required = {"text", "label", "split"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"dataset sem colunas: {', '.join(sorted(missing))}")
    subsets = {name: frame[frame["split"] == name].copy() for name in ("train", "validation", "test")}
    if any(subset.empty or subset["label"].nunique() != 2 for subset in subsets.values()):
        raise ValueError("cada split deve conter as duas classes")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"dataset": str(dataset_path), "seed": seed, "models": {}}
    for name, model in _models(seed).items():
        start = time.perf_counter()
        model.fit(subsets["train"]["text"], subsets["train"]["label"])
        fit_seconds = time.perf_counter() - start
        validation_start = time.perf_counter()
        validation_probabilities = model.predict_proba(subsets["validation"]["text"])[:, 1]
        validation_elapsed = time.perf_counter() - validation_start
        selected_threshold = select_threshold(
            subsets["validation"]["label"].to_numpy(), validation_probabilities
        )
        model_result: dict[str, object] = {
            "fit_seconds": fit_seconds,
            "threshold_selection": {
                "split": "validation",
                "objective": "f1_macro",
                "selected_threshold": selected_threshold,
            },
            "splits": {},
        }
        for split in ("validation", "test"):
            if split == "validation":
                probabilities, elapsed = validation_probabilities, validation_elapsed
            else:
                split_start = time.perf_counter()
                probabilities = model.predict_proba(subsets[split]["text"])[:, 1]
                elapsed = time.perf_counter() - split_start
            labels = subsets[split]["label"].to_numpy()
            model_result["splits"][split] = {
                "default_threshold": evaluate(labels, probabilities),
                "validation_selected_threshold": evaluate(labels, probabilities, selected_threshold),
                "latency_ms_per_message": elapsed * 1_000 / len(subsets[split]),
            }
            pd.DataFrame(
                {
                    "id": subsets[split]["id"],
                    "label": subsets[split]["label"],
                    "probability_textual_risk": probabilities,
                }
            ).to_csv(output / f"{name}_{split}_predictions.csv", index=False)
        artifact = output / f"{name}.joblib"
        joblib.dump(model, artifact)
        model_result["artifact"] = str(artifact)
        model_result["artifact_size_bytes"] = artifact.stat().st_size
        results["models"][name] = model_result
    (output / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
