"""Create a blinded, deterministic queue for human label review; it changes no labels."""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/evaluation/fakewhatsapp_external.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/fakewhatsapp_label_review.csv"))
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.size < 2:
        raise SystemExit("--size deve ser pelo menos 2")
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    if set(groups) != {"verdadeiro", "falso"}:
        raise SystemExit("a entrada deve conter verdadeiro e falso")
    chooser = random.Random(args.seed)
    selected: list[dict[str, str]] = []
    for label in ("verdadeiro", "falso"):
        values = sorted(groups[label], key=lambda row: row["id"])
        chooser.shuffle(values)
        selected.extend(values[: args.size // 2])
    selected = sorted(selected, key=lambda row: row["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("review_id", "original_id", "text", "dataset_label_blinded", "topic", "source", "published_at", "split", "review_status", "reviewer_label", "evidence_url", "reviewer_notes")
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            writer.writerow({"review_id": f"fw-review-{index:03d}", "original_id": row["id"], "text": row["text"],
                             "dataset_label_blinded": "", "topic": row["topic"], "source": row["source"],
                             "published_at": row["published_at"], "split": row["split"], "review_status": "pending",
                             "reviewer_label": "", "evidence_url": "", "reviewer_notes": ""})
    print({"records": len(selected), "output": str(args.output), "labels_blinded": True, "seed": args.seed})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
