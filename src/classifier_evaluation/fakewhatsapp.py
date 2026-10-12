"""Deterministic import of labeled FakeWhatsApp.BR messages for offline evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from .contracts import DatasetSplit, TruthLabel

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ImportSummary:
    raw_rows: int
    labeled_rows: int
    excluded_media_rows: int
    excluded_long_rows: int
    duplicate_rows: int
    conflicting_duplicate_rows: int
    calibration_candidates: int
    test_candidates: int
    calibration_exported: int
    test_exported: int
    overlapping_groups: int
    seed: int
    calibration_until: str
    test_from: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip().casefold()


def _published_at(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _stratified_sample(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= limit:
        return sorted(rows, key=lambda row: row["id"])
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    randomizer = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label in (TruthLabel.TRUE.value, TruthLabel.FALSE.value):
        candidates = sorted(grouped[label], key=lambda row: row["id"])
        randomizer.shuffle(candidates)
        quota = round(limit * len(candidates) / len(rows))
        quota = max(1, min(quota, len(candidates)))
        selected.extend(candidates[:quota])
    while len(selected) > limit:
        selected.pop()
    while len(selected) < limit:
        selected_ids = {row["id"] for row in selected}
        remainder = [row for row in sorted(rows, key=lambda row: row["id"]) if row["id"] not in selected_ids]
        selected.append(remainder[0])
    return sorted(selected, key=lambda row: row["id"])


def build_fakewhatsapp_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    calibration_until: date = date(2018, 9, 30),
    test_from: date = date(2018, 10, 1),
    max_calibration: int = 750,
    max_test: int = 1_000,
    max_text_chars: int = 5_000,
    seed: int = 42,
) -> ImportSummary:
    """Create an ignored external-evaluation CSV from existing human dataset labels.

    The source uses ``misinformation=1`` as false and ``0`` as true. Records with
    unknown label, media placeholders, duplicate text, or dates outside the temporal
    windows are excluded. Topic is intentionally a temporal cohort, not inferred text.
    """
    if calibration_until >= test_from:
        raise ValueError("calibration_until deve ser anterior a test_from")
    if max_calibration < 2 or max_test < 2 or max_text_chars < 5:
        raise ValueError("cada split deve permitir ao menos duas mensagens")

    raw_path = Path(input_path)
    candidates: list[dict[str, str]] = []
    excluded_media = 0
    excluded_long = 0
    labeled_rows = 0
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "text", "misinformation", "group", "url"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"colunas ausentes no FakeWhatsApp.BR: {', '.join(sorted(missing))}")
        raw_rows = 0
        for number, source_row in enumerate(reader, 2):
            raw_rows += 1
            label = source_row["misinformation"].strip()
            text = source_row["text"].strip()
            if label not in {"0", "1"}:
                continue
            labeled_rows += 1
            if not text or text == "<Arquivo de mídia oculto>":
                excluded_media += 1
                continue
            if len(text) > max_text_chars:
                excluded_long += 1
                continue
            published_at = _published_at(source_row["timestamp"].strip())
            split: DatasetSplit | None = None
            if published_at <= calibration_until:
                split = DatasetSplit.CALIBRATION
            elif published_at >= test_from:
                split = DatasetSplit.TEST
            if split is None:
                continue
            month = published_at.strftime("%Y-%m")
            group = source_row["group"].strip() or "unknown"
            original_url = source_row["url"].strip()
            source = f"FakeWhatsApp.BR_2018; group={group}; original_url={original_url}"
            candidates.append(
                {
                    "id": f"fakewhatsapp-{number}",
                    "text": text,
                    "label": TruthLabel.FALSE.value if label == "1" else TruthLabel.TRUE.value,
                    "topic": f"coorte_temporal:{month}",
                    "source": source,
                    "published_at": published_at.isoformat(),
                    "split": split.value,
                    "_normalized_text": _normalize_text(text),
                    "_group": group,
                }
            )

    by_text: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_text[row["_normalized_text"]].append(row)
    deduplicated: list[dict[str, str]] = []
    duplicate_rows = 0
    conflicting_rows = 0
    for duplicates in by_text.values():
        labels = {row["label"] for row in duplicates}
        if len(duplicates) == 1:
            deduplicated.extend(duplicates)
        elif len(labels) == 1:
            deduplicated.append(sorted(duplicates, key=lambda row: row["id"])[0])
            duplicate_rows += len(duplicates) - 1
        else:
            conflicting_rows += len(duplicates)

    calibration = [row for row in deduplicated if row["split"] == DatasetSplit.CALIBRATION.value]
    test = [row for row in deduplicated if row["split"] == DatasetSplit.TEST.value]
    exported_calibration = _stratified_sample(calibration, max_calibration, seed)
    exported_test = _stratified_sample(test, max_test, seed + 1)
    calibration_groups = {row["_group"] for row in calibration}
    test_groups = {row["_group"] for row in test}
    for row in [*exported_calibration, *exported_test]:
        row.pop("_normalized_text", None)
        row.pop("_group", None)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ("id", "text", "label", "topic", "source", "published_at", "split")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([*exported_calibration, *exported_test])

    summary = ImportSummary(
        raw_rows=raw_rows,
        labeled_rows=labeled_rows,
        excluded_media_rows=excluded_media,
        excluded_long_rows=excluded_long,
        duplicate_rows=duplicate_rows,
        conflicting_duplicate_rows=conflicting_rows,
        calibration_candidates=len(calibration),
        test_candidates=len(test),
        calibration_exported=len(exported_calibration),
        test_exported=len(exported_test),
        overlapping_groups=len(calibration_groups.intersection(test_groups)),
        seed=seed,
        calibration_until=calibration_until.isoformat(),
        test_from=test_from.isoformat(),
    )
    if manifest_path:
        manifest = {
            "dataset": "FakeWhatsApp.BR_2018",
            "input": str(raw_path),
            "output": str(output),
            "label_mapping": {"0": "verdadeiro", "1": "falso"},
            "topic_definition": "coorte temporal; não representa tema semântico",
            "limitations": [
                "split temporal reduz vazamento por tempo, mas grupos podem atravessar os splits",
                "rótulos e proveniência devem ser auditados antes de uma decisão de produção",
                "a amostra estratificada não preserva necessariamente a prevalência operacional",
            ],
            "summary": summary.as_dict(),
        }
        Path(manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
