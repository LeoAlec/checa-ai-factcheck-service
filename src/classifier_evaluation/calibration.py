"""Post-hoc calibration fitted only on the external calibration split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1e-7


def _check_inputs(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    labels = np.asarray(labels, dtype=int)
    if len(probabilities) != len(labels) or not len(labels):
        raise ValueError("probabilidades e rótulos devem ter o mesmo tamanho não vazio")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("calibração requer exemplos das classes verdadeiro=0 e falso=1")
    return probabilities, labels


def _logit(probabilities: np.ndarray) -> np.ndarray:
    return np.log(probabilities / (1 - probabilities))


@dataclass
class SigmoidCalibrator:
    """Platt scaling for the TF-IDF model's scalar score."""

    coefficient: float | None = None
    intercept: float | None = None
    method: str = "sigmoid_platt"

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "SigmoidCalibrator":
        probabilities, labels = _check_inputs(probabilities, labels)
        scores = _logit(probabilities)
        coefficient, intercept = 1.0, 0.0
        regularization = 1e-6
        for _ in range(100):
            values = np.clip(coefficient * scores + intercept, -35, 35)
            predictions = 1 / (1 + np.exp(-values))
            weights = np.maximum(predictions * (1 - predictions), 1e-9)
            gradient = np.array(
                [np.sum((predictions - labels) * scores) + regularization * coefficient,
                 np.sum(predictions - labels)]
            )
            hessian = np.array(
                [
                    [np.sum(weights * scores * scores) + regularization, np.sum(weights * scores)],
                    [np.sum(weights * scores), np.sum(weights) + regularization],
                ]
            )
            step = np.linalg.solve(hessian, gradient)
            coefficient -= float(step[0])
            intercept -= float(step[1])
            if np.linalg.norm(step) < 1e-8:
                break
        self.coefficient = coefficient
        self.intercept = intercept
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.coefficient is None or self.intercept is None:
            raise RuntimeError("calibrador ainda não foi ajustado")
        clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
        values = np.clip(self.coefficient * _logit(clipped) + self.intercept, -35, 35)
        return 1 / (1 + np.exp(-values))

    def parameters(self) -> dict[str, float]:
        if self.coefficient is None or self.intercept is None:
            return {}
        return {
            "coefficient": self.coefficient,
            "intercept": self.intercept,
        }


@dataclass
class TemperatureCalibrator:
    """Temperature scaling for BERTimbau's binary probability/logit."""

    temperature: float = 1.0
    method: str = "temperature_scaling"

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        probabilities, labels = _check_inputs(probabilities, labels)
        logits = _logit(probabilities)
        candidates = np.geomspace(0.05, 10.0, 600)
        losses = []
        for candidate in candidates:
            predictions = np.clip(1 / (1 + np.exp(-logits / candidate)), EPSILON, 1 - EPSILON)
            losses.append(float(-np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))))
        self.temperature = float(candidates[int(np.argmin(losses))])
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
        return 1 / (1 + np.exp(-_logit(clipped) / self.temperature))

    def parameters(self) -> dict[str, float]:
        return {"temperature": self.temperature}
