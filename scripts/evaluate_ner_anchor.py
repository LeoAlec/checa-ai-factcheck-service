"""Compare heuristic and spaCy entity-anchor safeguards on a labeled benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.rag.entities import all_entities_present, extract_entities, spacy_entities


def metrics(rows: list[dict], key: str) -> dict:
    correct = sum(row["expected_anchor"] == row[key] for row in rows)
    fp = sum(not row["expected_anchor"] and row[key] for row in rows)
    fn = sum(row["expected_anchor"] and not row[key] for row in rows)
    return {"accuracy": correct / len(rows), "false_positive": fp, "false_negative": fn}


def run(cases: list[dict]) -> dict:
    if spacy_entities("Teste") is None:
        raise RuntimeError("Modelo spaCy indisponível; instale pt_core_news_lg antes de avaliar.")
    rows = []
    timings = {"heuristic": 0.0, "spacy": 0.0}
    for case in cases:
        start = time.perf_counter()
        heuristic = all_entities_present(case["claim"], case["evidence"], backend="heuristic")
        heuristic_entities = extract_entities(case["claim"], backend="heuristic")
        timings["heuristic"] += time.perf_counter() - start
        start = time.perf_counter()
        spacy = all_entities_present(case["claim"], case["evidence"], backend="spacy")
        entities = extract_entities(case["claim"], backend="spacy")
        timings["spacy"] += time.perf_counter() - start
        rows.append({
            **case,
            "heuristic_anchor": heuristic,
            "spacy_anchor": spacy,
            "heuristic_entities": heuristic_entities,
            "spacy_entities": entities,
        })
    return {
        "n": len(rows),
        "heuristic": {**metrics(rows, "heuristic_anchor"), "mean_latency_ms": timings["heuristic"] * 1_000 / len(rows)},
        "spacy": {**metrics(rows, "spacy_anchor"), "mean_latency_ms": timings["spacy"] * 1_000 / len(rows)},
        "cases": rows,
        "scope": "Entity-anchor safeguard only; this does not evaluate retrieval or final LLM verdicts.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/eval/ner_anchor_benchmark_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/eval/ner_anchor_spacy_v1.json"))
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = run(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
