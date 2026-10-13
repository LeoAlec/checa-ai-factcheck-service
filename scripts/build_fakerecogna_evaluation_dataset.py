"""Build a controlled second external evaluation set from FakeRecogna.xlsx."""
from __future__ import annotations
import argparse
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier_evaluation.fakerecogna import build_fakerecogna_dataset

def _date(value: str) -> date: return date.fromisoformat(value)
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/FakeRecogna.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/fakerecogna_external.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/evaluation/fakerecogna_external.manifest.json"))
    parser.add_argument("--domain", action="append", default=["g1.globo.com"])
    parser.add_argument("--calibration-until", type=_date, default=date(2020, 12, 31))
    parser.add_argument("--test-from", type=_date, default=date(2021, 1, 1))
    parser.add_argument("--max-calibration", type=int, default=750)
    parser.add_argument("--max-test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(build_fakerecogna_dataset(args.input, args.output, manifest_path=args.manifest, domains=tuple(args.domain), calibration_until=args.calibration_until, test_from=args.test_from, max_calibration=args.max_calibration, max_test=args.max_test, seed=args.seed).as_dict())
    return 0
if __name__ == "__main__": raise SystemExit(main())
