"""Train word and character TF-IDF baselines for WhatsApp textual risk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.whatsapp_risk.training import train_baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/whatsapp_risk.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/whatsapp_risk_baselines"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = train_baselines(args.dataset, args.output, seed=args.seed)
    for name, result in results["models"].items():
        test = result["splits"]["test"]["default_threshold"]
        print(name, {key: test[key] for key in ("f1_macro", "f1_misinformation", "roc_auc", "pr_auc", "ece")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
