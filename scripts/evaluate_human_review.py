"""Create a quality-gate report from adjudicated human reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.human_evaluation.quality_gate import build_quality_report, load_cases, load_reviews, markdown_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--approve-with-exceptions-by",
        help="Identificador do responsável que aprova explicitamente métricas abaixo do limiar.",
    )
    args = parser.parse_args()

    cases = load_cases(args.responses)
    report = build_quality_report(
        cases,
        load_reviews(args.reviews, cases),
        approved_with_exceptions_by=args.approve_with_exceptions_by,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "human_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "human_quality_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"{report['decision']}: {args.output_dir / 'human_quality_report.md'}")
    return 0 if report["decision"] in {"approved_for_backend_integration", "approved_by_owner_with_exceptions"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
