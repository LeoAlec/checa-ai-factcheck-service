"""Evaluate BGE reranking on the frozen candidate set; intended for a Colab L4."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


def reciprocal_rank(urls: list[str], expected_url: str) -> float:
    for rank, url in enumerate(urls, 1):
        if url == expected_url:
            return 1.0 / rank
    return 0.0


def paired_bootstrap_delta(before: list[float], after: list[float], iterations: int = 10_000) -> dict:
    rng = random.Random(42)
    n = len(before)
    deltas = []
    for _ in range(iterations):
        indexes = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(after[i] - before[i] for i in indexes) / n)
    deltas.sort()
    return {
        "estimate": sum(a - b for a, b in zip(after, before)) / n,
        "low": deltas[int(iterations * 0.025)],
        "high": deltas[int(iterations * 0.975)],
    }


def exact_sign_p_value(improved: int, worsened: int) -> float:
    n = improved + worsened
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(improved, worsened) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def evaluate(rows: list[dict], model, batch_size: int) -> dict:
    positives = [row for row in rows if row["expected_status"] == "matched"]
    baseline_rr, reranked_rr, details = [], [], []
    started = time.perf_counter()
    for row in positives:
        candidates = row["candidates"]
        baseline_urls = [candidate["url"] for candidate in candidates]
        pairs = [(row["query"], f"{c['title']}. {c['text']}") for c in candidates]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False) if pairs else []
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
        reranked_urls = [candidate["url"] for candidate, _ in ranked]
        baseline_rr.append(reciprocal_rank(baseline_urls, row["expected_url"]))
        reranked_rr.append(reciprocal_rank(reranked_urls, row["expected_url"]))
        details.append(
            {
                "id": row["id"],
                "baseline_rank": next((i for i, url in enumerate(baseline_urls, 1) if url == row["expected_url"]), None),
                "reranked_rank": next((i for i, url in enumerate(reranked_urls, 1) if url == row["expected_url"]), None),
            }
        )
    elapsed = time.perf_counter() - started
    improved = sum(a < b for a, b in zip(baseline_rr, reranked_rr))
    worsened = sum(a > b for a, b in zip(baseline_rr, reranked_rr))
    return {
        "n": len(positives),
        "baseline_mrr": sum(baseline_rr) / len(baseline_rr),
        "reranked_mrr": sum(reranked_rr) / len(reranked_rr),
        "delta_mrr": paired_bootstrap_delta(baseline_rr, reranked_rr),
        "improved": improved,
        "worsened": worsened,
        "unchanged": sum(a == b for a, b in zip(baseline_rr, reranked_rr)),
        "exact_sign_test_p": exact_sign_p_value(improved, worsened),
        "elapsed_seconds": elapsed,
        "pairs_per_second": sum(len(row["candidates"]) for row in positives) / elapsed,
        "cases": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/eval/reranker_benchmark_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval/reranker_bge_v2_m3.json"))
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--revision", help="Commit SHA do modelo; se omitido, resolve e registra o SHA atual")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import torch
    from huggingface_hub import model_info
    from sentence_transformers import CrossEncoder

    revision = args.revision or model_info(args.model).sha
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.model, revision=revision, device=device, model_kwargs={"torch_dtype": "auto"})
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = evaluate(rows, model, args.batch_size)
    report.update({"model": args.model, "model_revision": revision, "device": device})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
