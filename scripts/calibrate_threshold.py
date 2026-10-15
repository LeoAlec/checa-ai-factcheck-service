"""Select, evaluate and persist a retrieval threshold without split leakage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.calibration import build_calibration_artifact  # noqa: E402

OUT = ROOT / "evals" / "calibrated_thresholds.json"
MIN_CASES = 40


def load_dataset_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def attach_splits(report_rows: list[dict[str, Any]], dataset_path: Path) -> list[dict[str, Any]]:
    metadata = load_dataset_metadata(dataset_path)
    enriched: list[dict[str, Any]] = []
    for row in report_rows:
        case_id = str(row.get("id", ""))
        if case_id not in metadata:
            raise ValueError(f"caso {case_id!r} não existe no dataset")
        split = metadata[case_id].get("split")
        if split not in {"calibration", "test"}:
            raise ValueError(f"caso {case_id!r} possui split inválido: {split!r}")
        enriched.append({**row, "split": split})
    return enriched


def current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibra limiar por provedor/modelo")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "retrieval" / "retrieval_eval_report.json")
    parser.add_argument("--dataset", type=Path, help="JSONL com os splits; padrão: eval_file do relatório")
    parser.add_argument("--strategy", choices=("max_f1", "min_precision"), default="min_precision")
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--promotion-tier", choices=("none", "pilot", "production"), default="none")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    dataset = args.dataset or Path(report["eval_file"])
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    rows = attach_splits(report["cases"], dataset)
    key = f"{report['embedding_provider']}/{report['embedding_model']}"
    result = build_calibration_artifact(
        rows=rows,
        dataset_path=dataset,
        provider=report["embedding_provider"],
        model=report["embedding_model"],
        collection=report["collection"],
        strategy=args.strategy,
        promotion_tier=args.promotion_tier,
        code_commit=current_commit(),
        min_precision=args.min_precision,
        minimum_cases=MIN_CASES,
        dataset_label=dataset.relative_to(ROOT).as_posix() if dataset.is_relative_to(ROOT) else dataset.as_posix(),
    )
    print(json.dumps({key: result}, indent=2, ensure_ascii=False))
    current = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    current[key] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Gravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
