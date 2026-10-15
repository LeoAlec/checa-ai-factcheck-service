import hashlib
import json
from pathlib import Path

import src.config as config
from src.evaluation.calibration import (
    artifact_is_eligible,
    build_calibration_artifact,
    select_threshold,
)


def _rows() -> list[dict]:
    return [
        {"id": "c1", "split": "calibration", "expected_status": "matched", "rank": 1, "best_score": 0.80},
        {"id": "c2", "split": "calibration", "expected_status": "abstained", "rank": None, "best_score": 0.70},
        {"id": "c3", "split": "calibration", "expected_status": "matched", "rank": 1, "best_score": 0.60},
        {"id": "t1", "split": "test", "expected_status": "abstained", "rank": None, "best_score": 0.99},
        {"id": "t2", "split": "test", "expected_status": "matched", "rank": 1, "best_score": 0.10},
    ]


def test_threshold_selection_never_uses_test_split() -> None:
    baseline = select_threshold(_rows(), strategy="max_f1")
    changed_test = [dict(row, best_score=0.01) if row["split"] == "test" else row for row in _rows()]
    assert select_threshold(changed_test, strategy="max_f1") == baseline


def test_min_precision_and_max_f1_are_explicit_strategies() -> None:
    max_f1 = select_threshold(_rows(), strategy="max_f1")
    conservative = select_threshold(_rows(), strategy="min_precision", min_precision=1.0)
    assert max_f1.strategy == "max_f1"
    assert conservative.strategy == "min_precision"
    assert conservative.precision == 1.0
    assert conservative.threshold >= max_f1.threshold


def test_artifact_records_provenance_and_split_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    artifact = build_calibration_artifact(
        rows=_rows(),
        dataset_path=dataset,
        provider="openai",
        model="embedding-model",
        collection="collection",
        strategy="max_f1",
        promotion_tier="pilot",
        code_commit="abc123",
    )
    assert artifact["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert artifact["n_calibration"] == 3
    assert artifact["n_test"] == 2
    assert artifact["calibration_metrics"]["n_cases"] == 3
    assert artifact["test_metrics"]["n_cases"] == 2


def test_production_rejects_unreliable_or_pilot_artifacts() -> None:
    base = {"threshold": 0.65, "reliable": True, "promotion_tier": "production"}
    assert artifact_is_eligible(base, runtime_tier="production")
    assert not artifact_is_eligible({**base, "reliable": False}, runtime_tier="production")
    assert not artifact_is_eligible({**base, "promotion_tier": "pilot"}, runtime_tier="production")
    assert artifact_is_eligible({**base, "reliable": False, "promotion_tier": "pilot"}, runtime_tier="pilot")


def test_artifact_is_json_serializable(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    artifact = build_calibration_artifact(
        rows=_rows(), dataset_path=dataset, provider="p", model="m", collection="c",
        strategy="max_f1", promotion_tier="pilot", code_commit=None,
    )
    json.dumps(artifact)


def test_runtime_only_loads_artifact_for_eligible_tier(tmp_path: Path, monkeypatch) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    key = f"{config.EMBEDDING_PROVIDER}/{config.EMBED_MODEL}"
    (evals / "calibrated_thresholds.json").write_text(
        json.dumps({key: {"threshold": 0.65, "reliable": False, "promotion_tier": "pilot"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("RETRIEVAL_CALIBRATION_TIER", "production")
    assert config._calibrated_threshold(0.60) == 0.60
    monkeypatch.setenv("RETRIEVAL_CALIBRATION_TIER", "pilot")
    assert config._calibrated_threshold(0.60) == 0.65
