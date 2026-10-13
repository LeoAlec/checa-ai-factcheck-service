"""Build a reproducible external-evaluation CSV from FakeWhatsApp.BR labels."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier_evaluation.fakewhatsapp import build_fakewhatsapp_dataset


def _date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/fakeWhatsApp.BR_2018.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/fakewhatsapp_external.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/evaluation/fakewhatsapp_external.manifest.json"))
    parser.add_argument("--calibration-until", type=_date, default=date(2018, 9, 30))
    parser.add_argument("--test-from", type=_date, default=date(2018, 10, 1))
    parser.add_argument("--max-calibration", type=int, default=750)
    parser.add_argument("--max-test", type=int, default=1_000)
    parser.add_argument("--max-text-chars", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = build_fakewhatsapp_dataset(
        args.input,
        args.output,
        manifest_path=args.manifest,
        calibration_until=args.calibration_until,
        test_from=args.test_from,
        max_calibration=args.max_calibration,
        max_test=args.max_test,
        max_text_chars=args.max_text_chars,
        seed=args.seed,
    )
    print(summary.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
