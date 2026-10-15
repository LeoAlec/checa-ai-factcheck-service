"""Build an offline reranker benchmark from cached dense results and the Silver corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def normalize_url(value: str) -> str:
    parts = urlsplit(value.strip())
    path = "/" + "/".join(segment for segment in parts.path.split("/") if segment)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path.rstrip("/"), "", ""))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_rows(benchmark: list[dict], report: dict, corpus: list[dict]) -> tuple[list[dict], list[str]]:
    benchmark_by_id = {row["id"]: row for row in benchmark}
    corpus_by_url = {normalize_url(row["url"]): row for row in corpus}
    output: list[dict] = []
    missing: set[str] = set()
    for result in report["cases"]:
        case = benchmark_by_id[result["id"]]
        candidates = []
        for rank, url in enumerate(result.get("ranked_urls", []), 1):
            document = corpus_by_url.get(normalize_url(url))
            if document is None:
                missing.add(url)
                continue
            candidates.append(
                {
                    "baseline_rank": rank,
                    "url": url,
                    "title": document.get("titulo", ""),
                    "text": document.get("texto", ""),
                }
            )
        output.append(
            {
                "id": result["id"],
                "query": case["query"],
                "split": case.get("split"),
                "expected_status": case["expected_status"],
                "expected_url": case.get("expected_url"),
                "candidates": candidates,
            }
        )
    return output, sorted(missing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/eval/retrieval_benchmark_120_paraphrased.jsonl"))
    parser.add_argument("--dense-report", type=Path, default=Path("reports/retrieval/para_dense.json"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("../checa-ai-ml/data/processed/fact_checks_silver.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/eval/reranker_benchmark_v1.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/retrieval/reranker_benchmark_v1_manifest.json"))
    args = parser.parse_args()

    rows, missing = build_rows(
        load_jsonl(args.benchmark),
        json.loads(args.dense_report.read_text(encoding="utf-8")),
        load_jsonl(args.corpus),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    matched = [row for row in rows if row["expected_status"] == "matched"]
    manifest = {
        "rows": len(rows),
        "matched_rows": len(matched),
        "rows_with_candidates": sum(bool(row["candidates"]) for row in rows),
        "matched_expected_url_in_candidates": sum(
            any(normalize_url(c["url"]) == normalize_url(row["expected_url"]) for c in row["candidates"])
            for row in matched
        ),
        "missing_candidate_urls": missing,
        "source_report": str(args.dense_report).replace("\\", "/"),
        "corpus": str(args.corpus).replace("\\", "/"),
        "note": "Candidate set is fixed; this benchmark measures reranking only, not retrieval recall or abstention.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
