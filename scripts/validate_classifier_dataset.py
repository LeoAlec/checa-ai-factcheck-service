"""Validate a manually curated external classifier-evaluation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier_evaluation.dataset import DatasetValidationError, dataset_summary, load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        records = load_dataset(args.dataset)
    except DatasetValidationError as exc:
        print(f"Dataset inválido: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dataset_summary(records), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
