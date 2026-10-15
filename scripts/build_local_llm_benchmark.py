"""Materialize a development-only evidence-verdict benchmark for local LLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "id": item["id"],
            "claim": item["claim"],
            "evidence_title": item["evidence"]["titulo"],
            "evidence_text": item["evidence"]["texto"],
            "gold_conclusion": item["gold_conclusion"],
            "tags": item["tags"],
        }
        for item in items
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evals/golden_claims.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/local_llm_verdict_dev_v1.jsonl"))
    args = parser.parse_args()
    items = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = build_rows(items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output), "split": "development_only"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
