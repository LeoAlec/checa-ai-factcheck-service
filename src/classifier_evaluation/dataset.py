"""Load and validate externally curated short-message datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import ValidationError

from .contracts import DatasetRecord, DatasetSplit, TruthLabel

DATASET_COLUMNS = ("id", "text", "label", "topic", "source", "published_at", "split")


class DatasetValidationError(ValueError):
    pass


def _raw_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(DATASET_COLUMNS).difference(reader.fieldnames or [])
            if missing:
                raise DatasetValidationError(
                    f"colunas obrigatórias ausentes: {', '.join(sorted(missing))}"
                )
            return list(reader)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"JSON inválido na linha {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise DatasetValidationError(f"linha {line_number} não contém um objeto JSON")
            rows.append(value)
        return rows
    raise DatasetValidationError("formato não suportado; use CSV ou JSONL")


def load_dataset(path: str | Path, *, require_evaluation_ready: bool = True) -> list[DatasetRecord]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetValidationError(f"dataset não encontrado: {dataset_path}")
    raw = _raw_rows(dataset_path)
    if not raw:
        raise DatasetValidationError("dataset vazio; o template precisa ser curado externamente")

    records: list[DatasetRecord] = []
    errors: list[str] = []
    for index, row in enumerate(raw, 2 if dataset_path.suffix.lower() == ".csv" else 1):
        try:
            records.append(DatasetRecord.model_validate(row))
        except ValidationError as exc:
            errors.append(f"linha {index}: {exc.errors(include_url=False)}")
    if errors:
        raise DatasetValidationError("; ".join(errors))

    ids = [record.id for record in records]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise DatasetValidationError(f"ids duplicados: {', '.join(duplicates)}")

    if require_evaluation_ready:
        for split in DatasetSplit:
            subset = [record for record in records if record.split is split]
            if not subset:
                raise DatasetValidationError(f"split obrigatório vazio ou ausente: {split.value}")
            labels = {record.label for record in subset}
            if labels != set(TruthLabel):
                raise DatasetValidationError(
                    f"split {split.value} deve conter as duas classes: verdadeiro e falso"
                )
    return records


def dataset_summary(records: list[DatasetRecord]) -> dict[str, object]:
    return {
        "records": len(records),
        "splits": {
            split.value: sum(record.split is split for record in records) for split in DatasetSplit
        },
        "labels": {
            label.value: sum(record.label is label for record in records) for label in TruthLabel
        },
        "topics": sorted({record.topic for record in records}),
    }
