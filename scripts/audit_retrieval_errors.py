"""Export or validate human review of retrieval false positives."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.audit import (  # noqa: E402
    build_false_positive_audit,
    validate_review_rows,
    write_audit_csv,
)


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["id"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="padrão: eval_file do relatório")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--split", choices=("calibration", "test", "all"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval" / "false_positive_review.csv")
    parser.add_argument("--validate", type=Path, help="valida um CSV de revisão já preenchido")
    args = parser.parse_args()

    if args.validate:
        with args.validate.open(encoding="utf-8", newline="") as handle:
            validate_review_rows(csv.DictReader(handle))
        print(f"Revisão válida: {args.validate}")
        return 0

    report = json.loads(args.report.read_text(encoding="utf-8"))
    dataset = args.dataset or Path(report["eval_file"])
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    threshold = args.threshold if args.threshold is not None else float(report.get("current_threshold", 0.65))
    rows = build_false_positive_audit(
        report["cases"], metadata=load_metadata(dataset), threshold=threshold,
        split=None if args.split == "all" else args.split,
    )
    write_audit_csv(rows, args.output)
    print(f"Falsos positivos para revisão: {len(rows)} | arquivo: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
