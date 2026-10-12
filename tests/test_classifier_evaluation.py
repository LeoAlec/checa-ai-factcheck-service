from __future__ import annotations

import csv
import contextlib
import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.classifier_evaluation.adapters import BertimbauAdapter, TfidfAdapter
from src.classifier_evaluation.artifacts import inspect_artifact, inventory
from src.classifier_evaluation.contracts import ClassifierOutput, DatasetRecord
from src.classifier_evaluation.dataset import DatasetValidationError, load_dataset
from src.classifier_evaluation.fakewhatsapp import build_fakewhatsapp_dataset
from src.classifier_evaluation.fakerecogna import build_fakerecogna_dataset
from src.classifier_evaluation.metrics import classification_metrics, expected_calibration_error
from src.classifier_evaluation.report import build_report, write_report
from src.classifier_evaluation.runner import evaluate_model


class SerializableModel:
    classes_ = np.array([0, 1])

    def predict(self, texts):
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, texts):
        false = np.array([0.8 if "fals" in text else 0.2 for text in texts])
        return np.column_stack([1 - false, false])


class DeterministicAdapter:
    model_name = "tfidf"
    model_version = "test-v1"
    limitations = ["fixture sintética; não é resultado de avaliação"]

    def predict_probability_false(self, texts):
        return np.array([0.8 if "fals" in text else 0.2 for text in texts])


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def to(self, _device):
        return self

    def __getitem__(self, item):
        return FakeTensor(self.values[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert Path(path).is_dir()
        assert local_files_only is True
        return cls()

    def __call__(self, batch, **_kwargs):
        return {"input_ids": FakeTensor([[1, 2]] * len(batch))}


class FakeTransformerModel:
    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert Path(path).is_dir()
        assert local_files_only is True
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return None

    def __call__(self, **encoded):
        size = len(encoded["input_ids"].values)
        return types.SimpleNamespace(logits=FakeTensor([[0.0, 2.0]] * size))


def _rows() -> list[dict[str, str]]:
    return [
        {
            "id": f"id-{index}",
            "text": "mensagem falsa verificada" if index % 2 else "mensagem verdadeira verificada",
            "label": "falso" if index % 2 else "verdadeiro",
            "topic": "saude" if index < 5 else "politica",
            "source": f"https://example.org/check/{index}",
            "published_at": "2025-01-01",
            "split": "calibration" if index <= 4 else "test",
        }
        for index in range(1, 9)
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_load_tfidf_artifact_and_probability(tmp_path: Path):
    model_path = tmp_path / "best_model.joblib"
    joblib.dump(SerializableModel(), model_path)
    artifact = inspect_artifact("tfidf_model", model_path, "joblib")

    adapter = TfidfAdapter(model_path, artifact.version or "unknown")

    assert artifact.available is True
    assert artifact.size_bytes and artifact.size_bytes > 0
    assert artifact.version and artifact.version.startswith("sha256:")
    assert adapter.predict_probability_false(["isso é falso", "isso procede"]).tolist() == [0.8, 0.2]


def test_inventory_reports_missing_artifacts(tmp_path: Path):
    items = inventory(tmp_path / "missing.joblib", tmp_path / "missing-model")
    assert len(items) == 6
    assert all(item.available is False for item in items)


def test_bertimbau_rejects_incomplete_directory_before_optional_import(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        BertimbauAdapter(tmp_path, "test")


def test_load_bertimbau_artifact_from_local_directory(tmp_path: Path, monkeypatch):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checa_ai_config.json").write_text('{"max_length": 128}', encoding="utf-8")
    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = contextlib.nullcontext
    fake_torch.softmax = lambda tensor, dim: FakeTensor(
        np.exp(tensor.values) / np.exp(tensor.values).sum(axis=dim, keepdims=True)
    )
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.AutoModelForSequenceClassification = FakeTransformerModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    adapter = BertimbauAdapter(tmp_path, "test-v1", batch_size=2)
    probabilities = adapter.predict_probability_false(["mensagem um", "mensagem dois"])

    assert adapter.max_length == 128
    assert probabilities.tolist() == pytest.approx([0.880797, 0.880797])


def test_external_dataset_validation_accepts_separate_splits(tmp_path: Path):
    path = tmp_path / "external.csv"
    _write_csv(path, _rows())

    records = load_dataset(path)

    assert len(records) == 8
    assert {record.binary_label for record in records} == {0, 1}


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda rows: rows + [dict(rows[0])], "ids duplicados"),
        (lambda rows: [dict(row, split="calibration") for row in rows], "split obrigatório"),
        (
            lambda rows: [dict(row, label="verdadeiro") if row["split"] == "test" else row for row in rows],
            "deve conter as duas classes",
        ),
    ],
)
def test_external_dataset_validation_rejects_invalid_partitions(tmp_path: Path, mutation, expected):
    path = tmp_path / "external.csv"
    _write_csv(path, mutation(_rows()))
    with pytest.raises(DatasetValidationError, match=expected):
        load_dataset(path)


def test_metrics_include_probability_quality_and_confusion_matrix():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.7, 0.8, 0.4]

    metrics = classification_metrics(labels, probabilities)

    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["brier_score"] == pytest.approx(0.225)
    assert metrics["ece"] == pytest.approx(expected_calibration_error(labels, probabilities))
    assert metrics["reliability_curve"]


def test_evaluation_uses_calibration_and_test_separately():
    records = [DatasetRecord.model_validate(row) for row in _rows()]
    artifact = inspect_artifact("tfidf_model", None, "joblib")

    result = evaluate_model(DeterministicAdapter(), artifact, records)

    assert result["status"] == "evaluated"
    assert result["dataset"] == {
        "calibration_records": 4,
        "test_records": 4,
        "training_records_used": 0,
    }
    assert result["calibration"]["method"] == "sigmoid_platt"
    assert result["calibration"]["fit_split"] == "calibration"
    assert result["calibration"]["evaluation_split"] == "test"
    assert result["metrics"]["calibrated"]["accuracy"] == 1.0
    assert set(result["metrics"]["by_topic"]) == {"politica"}


def test_report_generation_without_real_results_recommends_no_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = inspect_artifact("tfidf_model", None, "joblib")
    report = build_report(
        artifact_inventory=[artifact.as_dict()],
        dataset={"status": "unavailable", "reason": "template vazio"},
        results=[
            {
                "status": "unavailable",
                "model_name": "tfidf",
                "artifact": artifact.as_dict(),
                "reason": "artefato não encontrado",
                "metrics": None,
            }
        ],
    )
    monkeypatch.setattr("src.classifier_evaluation.report._write_reliability_plot", lambda *_: None)

    json_path, markdown_path = write_report(report, tmp_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["recommendation"]["decision"] == "nao_integrar_nenhum_por_enquanto"
    assert "Nenhum resultado real" in markdown_path.read_text(encoding="utf-8")


def test_future_output_contract_assigns_threshold_band():
    output = ClassifierOutput.from_probability(
        model_name="tfidf",
        model_version="sha256:abc",
        probability_false=0.75,
        calibrated=True,
        limitations=["não verifica fatos"],
    )
    assert output.model_dump(mode="json") == {
        "model_name": "tfidf",
        "model_version": "sha256:abc",
        "probability_false": 0.75,
        "calibrated": True,
        "threshold_band": "alto",
        "limitations": ["não verifica fatos"],
    }


def test_fakewhatsapp_import_uses_existing_labels_temporal_split_and_deduplication(tmp_path: Path):
    source_path = tmp_path / "fakewhatsapp.csv"
    output_path = tmp_path / "external.csv"
    manifest_path = tmp_path / "manifest.json"
    source_rows = [
        {"timestamp": "2018-09-01 10:00:00", "text": "alegação verdadeira", "misinformation": "0", "group": "a", "url": "0"},
        {"timestamp": "2018-09-02 10:00:00", "text": "alegação falsa", "misinformation": "1", "group": "a", "url": "0"},
        {"timestamp": "2018-10-01 10:00:00", "text": "teste verdadeiro", "misinformation": "0", "group": "b", "url": "1"},
        {"timestamp": "2018-10-02 10:00:00", "text": "teste falso", "misinformation": "1", "group": "b", "url": "1"},
        {"timestamp": "2018-10-03 10:00:00", "text": "teste falso", "misinformation": "1", "group": "b", "url": "1"},
        {"timestamp": "2018-10-04 10:00:00", "text": "ignorar", "misinformation": "-1", "group": "b", "url": "1"},
        {"timestamp": "2018-10-05 10:00:00", "text": "x" * 5_001, "misinformation": "1", "group": "b", "url": "1"},
    ]
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_rows[0].keys())
        writer.writeheader()
        writer.writerows(source_rows)

    summary = build_fakewhatsapp_dataset(
        source_path,
        output_path,
        manifest_path=manifest_path,
        max_calibration=2,
        max_test=2,
    )
    records = load_dataset(output_path)

    assert summary.labeled_rows == 6
    assert summary.duplicate_rows == 1
    assert summary.excluded_long_rows == 1
    assert {record.split.value for record in records} == {"calibration", "test"}
    assert {record.label.value for record in records} == {"verdadeiro", "falso"}
    assert all(record.topic.startswith("coorte_temporal:") for record in records)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["dataset"] == "FakeWhatsApp.BR_2018"


def test_fakerecogna_import_uses_title_only_and_mixed_domain_temporal_split(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    source_path = tmp_path / "FakeRecogna.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(("Titulo", "Subtitulo", "Noticia", "Categoria", "Data", "Autor", "URL", "Classe"))
    sheet.append(("alegação falsa antiga", "", "explicação que não pode vazar", "saude", "01/10/2020", "", "https://g1.globo.com/a", 0))
    sheet.append(("alegação verdadeira antiga", "", "explicação", "saude", "02/10/2020", "", "https://g1.globo.com/b", 1))
    sheet.append(("alegação falsa nova", "", "explicação", "politica", "01/02/2021", "", "https://g1.globo.com/c", 0))
    sheet.append(("alegação verdadeira nova", "", "explicação", "politica", "02/02/2021", "", "https://g1.globo.com/d", 1))
    sheet.append(("fora do domínio", "", "", "saude", "02/02/2021", "", "https://boatos.org/e", 0))
    workbook.save(source_path)
    output_path, manifest_path = tmp_path / "external.csv", tmp_path / "manifest.json"

    summary = build_fakerecogna_dataset(source_path, output_path, manifest_path=manifest_path, max_calibration=2, max_test=2)
    records = load_dataset(output_path)

    assert len(records) == 4
    assert summary.excluded_non_target_domain == 1
    assert all("explicação" not in record.text for record in records)
    assert {record.label.value for record in records} == {"verdadeiro", "falso"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_field"].startswith("Titulo")
