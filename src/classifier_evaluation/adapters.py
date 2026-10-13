"""Read-only adapters for the candidate TF-IDF and BERTimbau artifacts."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Protocol, Sequence

import joblib
import numpy as np

from .legacy_preprocessing import PortugueseTextPreprocessor


class ClassifierAdapter(Protocol):
    model_name: str
    model_version: str
    limitations: list[str]

    def predict_probability_false(self, texts: Sequence[str]) -> np.ndarray: ...


class TfidfAdapter:
    model_name = "tfidf"

    def __init__(self, path: str | Path, model_version: str) -> None:
        self.path = Path(path)
        _register_legacy_preprocessor()
        try:
            self.model = joblib.load(self.path)
        except Exception as exc:
            raise RuntimeError(
                "não foi possível carregar o joblib; instale as mesmas dependências do projeto "
                "de treinamento (incluindo checa_ai, se referenciado pelo pipeline)"
            ) from exc
        if not hasattr(self.model, "predict"):
            raise TypeError("artefato TF-IDF não implementa predict")
        self.model_version = model_version
        self.limitations = [
            "classificador de padrão textual; não verifica fatos nem substitui evidências",
            "desempenho pode mudar fora do domínio e período do treino",
        ]

    def predict_probability_false(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(list(texts)), dtype=float)
            classes = list(getattr(self.model, "classes_", []))
            if 1 not in classes:
                raise ValueError("classes_ do modelo não contém o rótulo falso=1")
            result = probabilities[:, classes.index(1)]
        elif hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(list(texts)), dtype=float)
            classes = list(getattr(self.model, "classes_", [0, 1]))
            if len(classes) != 2 or 1 not in classes:
                raise ValueError("classes_ do modelo binário não contém o rótulo falso=1")
            positive_probability = 1.0 / (1.0 + np.exp(-scores))
            result = positive_probability if classes[1] == 1 else 1 - positive_probability
        else:
            raise TypeError("artefato TF-IDF não expõe predict_proba nem decision_function")
        return _validate_probabilities(result)


def _register_legacy_preprocessor() -> None:
    """Make the historical pickle import path available before ``joblib.load``."""
    package = sys.modules.setdefault("checa_ai", types.ModuleType("checa_ai"))
    module = sys.modules.setdefault("checa_ai.preprocessing", types.ModuleType("checa_ai.preprocessing"))
    module.PortugueseTextPreprocessor = PortugueseTextPreprocessor
    setattr(package, "preprocessing", module)


class BertimbauAdapter:
    model_name = "bertimbau"

    def __init__(
        self,
        path: str | Path,
        model_version: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.path = Path(path)
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"config.json ausente em {self.path}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "BERTimbau requer torch e transformers funcionais; verifique também se a política "
                "do sistema permite carregar as DLLs nativas do PyTorch"
            ) from exc

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        extra_config = self.path / "checa_ai_config.json"
        settings = json.loads(extra_config.read_text(encoding="utf-8")) if extra_config.exists() else {}
        self.max_length = int(settings.get("max_length", 256))
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.path, local_files_only=True
        ).to(device)
        self.model.eval()
        self.model_version = model_version
        self.limitations = [
            "classificador de padrão textual; não verifica fatos nem substitui evidências",
            "mais custoso em memória e latência que um baseline linear",
            "desempenho pode mudar fora do domínio e período do treino",
        ]

    def predict_probability_false(self, texts: Sequence[str]) -> np.ndarray:
        probabilities: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
                batch_probabilities = self._torch.softmax(logits, dim=-1)[:, 1]
            probabilities.append(batch_probabilities.detach().cpu().numpy())
        result = np.concatenate(probabilities) if probabilities else np.array([], dtype=float)
        return _validate_probabilities(result)


def _validate_probabilities(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(flattened)) or np.any((flattened < 0) | (flattened > 1)):
        raise ValueError("modelo produziu probabilidades inválidas")
    return flattened
