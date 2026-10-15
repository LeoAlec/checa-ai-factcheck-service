from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.whatsapp_risk.dataset import message_group_id, normalize_message, prepare_dataset
from src.whatsapp_risk.training import evaluate, select_threshold


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ("timestamp", "text", "misinformation", "midia", "url", "viral", "shares")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_normalize_message_masks_urls_and_whitespace() -> None:
    assert normalize_message(" Veja  HTTPS://x.test/a\nAGORA ") == "veja URL agora"
    assert message_group_id("A https://x.test") == message_group_id("a https://y.test")


def test_prepare_dataset_excludes_cross_period_duplicates_and_unknown_labels(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    base = {"midia": "0", "url": "0", "viral": "1", "shares": "2"}
    rows = [
        {**base, "timestamp": "2018-08-01 10:00:00", "text": "Mensagem repetida aqui", "misinformation": "1"},
        {**base, "timestamp": "2018-10-01 10:00:00", "text": "Mensagem repetida aqui", "misinformation": "1"},
        {**base, "timestamp": "2018-08-02 10:00:00", "text": "Mensagem verdadeira treino", "misinformation": "0"},
        {**base, "timestamp": "2018-09-02 10:00:00", "text": "Mensagem falsa validacao", "misinformation": "1"},
        {**base, "timestamp": "2018-10-02 10:00:00", "text": "Mensagem verdadeira teste", "misinformation": "0"},
        {**base, "timestamp": "2018-10-03 10:00:00", "text": "Mensagem sem rotulo aqui", "misinformation": "-1"},
    ]
    _write_source(source, rows)
    output, manifest = tmp_path / "dataset.csv", tmp_path / "manifest.json"

    summary = prepare_dataset(source, output, manifest)

    assert summary.labeled_rows == 5
    assert summary.cross_period_groups_excluded == 1
    assert summary.exported_rows == 3
    assert json.loads(manifest.read_text(encoding="utf-8"))["excluded_label"].startswith("-1")


def test_evaluate_reports_expected_binary_metrics() -> None:
    result = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert result["f1_macro"] == 1.0
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_threshold_selection_uses_supplied_validation_labels() -> None:
    threshold = select_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert 0.2 < threshold <= 0.3
