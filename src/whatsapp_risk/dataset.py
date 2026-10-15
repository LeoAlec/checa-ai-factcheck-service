"""Prepare leakage-resistant FakeWhatsApp.BR train/validation/test splits."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MEDIA_PLACEHOLDER = "<Arquivo de mídia oculto>"


@dataclass(frozen=True)
class DatasetManifest:
    raw_rows: int
    labeled_rows: int
    unique_labeled_texts: int
    exported_rows: int
    conflicting_groups: int
    cross_period_groups_excluded: int
    excluded_media_rows: int
    excluded_short_rows: int
    excluded_long_rows: int
    split_counts: dict[str, int]
    label_counts: dict[str, dict[str, int]]
    input_sha256: str
    train_until: str
    validation_until: str
    test_from: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_message(text: str) -> str:
    """Normalize superficial forwarding variation while retaining lexical content."""
    value = unicodedata.normalize("NFKC", text).casefold()
    value = _URL.sub(" URL ", value)
    return _WHITESPACE.sub(" ", value).strip()


def message_group_id(text: str) -> str:
    return hashlib.sha256(normalize_message(text).encode("utf-8")).hexdigest()[:20]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def _period(timestamp: datetime, train_until: date, validation_until: date, test_from: date) -> str | None:
    current = timestamp.date()
    if current <= train_until:
        return "train"
    if current <= validation_until:
        return "validation"
    if current >= test_from:
        return "test"
    return None


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    train_until: date = date(2018, 8, 31),
    validation_until: date = date(2018, 9, 30),
    test_from: date = date(2018, 10, 1),
    min_words: int = 3,
    max_chars: int = 10_000,
) -> DatasetManifest:
    """Export one representative per normalized text with strict temporal isolation.

    Groups occurring in more than one temporal period are excluded instead of being
    allowed to leak a repeated viral message into a later split.
    """
    if not train_until < validation_until < test_from:
        raise ValueError("datas devem obedecer train_until < validation_until < test_from")
    if min_words < 1 or max_chars < 1:
        raise ValueError("limites de texto devem ser positivos")

    source = Path(input_path)
    raw_rows = labeled_rows = 0
    excluded_media = excluded_short = excluded_long = 0
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "text", "misinformation", "midia", "url", "viral", "shares"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"colunas ausentes: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, 2):
            raw_rows += 1
            raw_label = row["misinformation"].strip()
            if raw_label not in {"0", "1"}:
                continue
            labeled_rows += 1
            text = row["text"].strip()
            if row["midia"].strip() == "1" or not text or text == _MEDIA_PLACEHOLDER:
                excluded_media += 1
                continue
            words = len(text.split())
            if words < min_words:
                excluded_short += 1
                continue
            if len(text) > max_chars:
                excluded_long += 1
                continue
            timestamp = _parse_timestamp(row["timestamp"])
            split = _period(timestamp, train_until, validation_until, test_from)
            if split is None:
                continue
            group_id = message_group_id(text)
            grouped[group_id].append(
                {
                    "id": f"fakewhatsapp-{row_number}",
                    "group_id": group_id,
                    "timestamp": timestamp,
                    "text": text,
                    "label": int(raw_label),
                    "split": split,
                    "has_url": int(row["url"] or 0),
                    "viral": int(row["viral"] or 0),
                    "shares": int(float(row["shares"] or 0)),
                }
            )

    exported: list[dict[str, object]] = []
    conflicting_groups = cross_period_groups = 0
    for records in grouped.values():
        labels = {int(record["label"]) for record in records}
        if len(labels) != 1:
            conflicting_groups += 1
            continue
        splits = {str(record["split"]) for record in records}
        if len(splits) != 1:
            cross_period_groups += 1
            continue
        representative = min(records, key=lambda record: (record["timestamp"], record["id"]))
        exported.append(representative)

    exported.sort(key=lambda record: (record["timestamp"], record["id"]))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ("id", "group_id", "timestamp", "text", "label", "split", "has_url", "viral", "shares")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in exported:
            serialized = dict(record)
            serialized["timestamp"] = record["timestamp"].isoformat()
            writer.writerow(serialized)

    split_counts = Counter(str(record["split"]) for record in exported)
    label_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        counts = Counter(str(record["label"]) for record in exported if record["split"] == split)
        label_counts[split] = {"0": counts["0"], "1": counts["1"]}
    manifest = DatasetManifest(
        raw_rows=raw_rows,
        labeled_rows=labeled_rows,
        unique_labeled_texts=len(grouped),
        exported_rows=len(exported),
        conflicting_groups=conflicting_groups,
        cross_period_groups_excluded=cross_period_groups,
        excluded_media_rows=excluded_media,
        excluded_short_rows=excluded_short,
        excluded_long_rows=excluded_long,
        split_counts={split: split_counts[split] for split in ("train", "validation", "test")},
        label_counts=label_counts,
        input_sha256=_sha256(source),
        train_until=train_until.isoformat(),
        validation_until=validation_until.isoformat(),
        test_from=test_from.isoformat(),
    )
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(
            {
                "dataset": "FakeWhatsApp.BR_2018",
                "task": "whatsapp_textual_misinformation_risk",
                "unit": "unique normalized textual message",
                "label_mapping": {"0": "non_misinformation", "1": "misinformation"},
                "selection_bias": "manual labels were assigned to viral messages and propagated to exact occurrences",
                "excluded_label": "-1 means unlabeled and is never treated as negative",
                "manifest": manifest.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest
