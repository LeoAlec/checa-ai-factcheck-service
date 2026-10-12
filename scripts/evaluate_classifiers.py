"""Evaluate external TF-IDF and BERTimbau artifacts without training them."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.classifier_evaluation.adapters import BertimbauAdapter, TfidfAdapter
from src.classifier_evaluation.artifacts import ArtifactInfo, inventory
from src.classifier_evaluation.dataset import DatasetValidationError, dataset_summary, load_dataset
from src.classifier_evaluation.report import build_report, write_report
from src.classifier_evaluation.runner import evaluate_model, unavailable_result


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _model_info(items: list[ArtifactInfo], name: str) -> ArtifactInfo:
    return next(item for item in items if item.name == name)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_path(os.getenv("CLASSIFIER_EXTERNAL_DATASET_PATH")))
    parser.add_argument("--tfidf-model", type=Path, default=_path(os.getenv("CLASSIFIER_TFIDF_MODEL_PATH")))
    parser.add_argument("--bertimbau-model", type=Path, default=_path(os.getenv("CLASSIFIER_BERTIMBAU_MODEL_PATH")))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("CLASSIFIER_EVALUATION_OUTPUT_DIR", "reports/classifier-evaluation")),
    )
    parser.add_argument("--device", default=os.getenv("CLASSIFIER_BERTIMBAU_DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("CLASSIFIER_BERTIMBAU_BATCH_SIZE", "16")))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("CLASSIFIER_DECISION_THRESHOLD", "0.5")))
    args = parser.parse_args()

    artifacts = inventory(
        args.tfidf_model,
        args.bertimbau_model,
        tfidf_metrics_path=_path(os.getenv("CLASSIFIER_TFIDF_METRICS_PATH")),
        tfidf_predictions_path=_path(os.getenv("CLASSIFIER_TFIDF_PREDICTIONS_PATH")),
        bertimbau_metrics_path=_path(os.getenv("CLASSIFIER_BERTIMBAU_METRICS_PATH")),
        bertimbau_predictions_path=_path(os.getenv("CLASSIFIER_BERTIMBAU_PREDICTIONS_PATH")),
    )

    records = None
    dataset_info: dict[str, object]
    try:
        if args.dataset is None:
            raise DatasetValidationError("CLASSIFIER_EXTERNAL_DATASET_PATH não configurado")
        records = load_dataset(args.dataset)
        dataset_info = {"status": "validated", "path": str(args.dataset.resolve()), **dataset_summary(records)}
    except DatasetValidationError as exc:
        dataset_info = {
            "status": "unavailable",
            "path": str(args.dataset.resolve()) if args.dataset else "",
            "reason": str(exc),
        }

    results = []
    for model_name, artifact_name, model_path in (
        ("tfidf", "tfidf_model", args.tfidf_model),
        ("bertimbau", "bertimbau_model", args.bertimbau_model),
    ):
        artifact = _model_info(artifacts, artifact_name)
        if not artifact.available:
            results.append(unavailable_result(model_name, artifact, "artefato não encontrado"))
            continue
        if records is None:
            results.append(unavailable_result(model_name, artifact, "dataset externo não está pronto"))
            continue
        try:
            started = perf_counter()
            adapter = (
                TfidfAdapter(model_path, artifact.version or "unknown")
                if model_name == "tfidf"
                else BertimbauAdapter(
                    model_path,
                    artifact.version or "unknown",
                    device=args.device,
                    batch_size=args.batch_size,
                )
            )
            load_ms = (perf_counter() - started) * 1_000
            result = evaluate_model(adapter, artifact, records, threshold=args.threshold)
            result["latency"]["model_load_ms"] = load_ms
            results.append(result)
        except Exception as exc:
            results.append(unavailable_result(model_name, artifact, f"falha de avaliação: {exc}"))

    report = build_report(
        artifact_inventory=[artifact.as_dict() for artifact in artifacts],
        dataset=dataset_info,
        results=results,
    )
    json_path, markdown_path = write_report(report, args.output_dir)
    print(f"JSON: {json_path.resolve()}")
    print(f"Markdown: {markdown_path.resolve()}")
    print(f"Recomendação: {report['recommendation']['decision']}")
    return 0 if records is not None and any(r["status"] == "evaluated" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
