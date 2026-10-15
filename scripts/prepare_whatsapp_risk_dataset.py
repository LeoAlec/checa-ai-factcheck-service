"""Prepare leakage-resistant FakeWhatsApp.BR splits for model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.whatsapp_risk.dataset import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/fakeWhatsApp.BR_2018.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/whatsapp_risk.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/whatsapp_risk.manifest.json"))
    args = parser.parse_args()
    print(prepare_dataset(args.input, args.output, args.manifest).as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
