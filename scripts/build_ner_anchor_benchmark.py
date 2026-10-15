"""Build a deterministic entity-anchor benchmark from reviewed golden evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POSITIVE_IDS = [
    "gc-018", "gc-019", "gc-020", "gc-021", "gc-022", "gc-026",
    "gc-027", "gc-028", "gc-029", "gc-030", "gc-065", "gc-066",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_cases(items: list[dict]) -> list[dict]:
    by_id = {item["id"]: item for item in items}
    missing = [case_id for case_id in POSITIVE_IDS if case_id not in by_id]
    if missing:
        raise ValueError(f"IDs ausentes: {missing}")

    cases = []
    for index, case_id in enumerate(POSITIVE_IDS):
        item = by_id[case_id]
        cases.append(
            {
                "id": f"anchor-positive-{case_id}",
                "claim": item["claim"],
                "evidence": item["evidence"]["titulo"] + ". " + item["evidence"]["texto"],
                "expected_anchor": True,
                "kind": "same_entity",
                "source_id": case_id,
            }
        )
        mismatch = by_id[POSITIVE_IDS[(index + 1) % len(POSITIVE_IDS)]]
        cases.append(
            {
                "id": f"anchor-negative-{case_id}",
                "claim": item["claim"],
                "evidence": mismatch["evidence"]["titulo"] + ". " + mismatch["evidence"]["texto"],
                "expected_anchor": False,
                "kind": "different_entity",
                "source_id": mismatch["id"],
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evals/golden_claims.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/ner_anchor_benchmark_v1.jsonl"))
    args = parser.parse_args()
    cases = build_cases(load_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "positive": len(POSITIVE_IDS), "negative": len(POSITIVE_IDS), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
