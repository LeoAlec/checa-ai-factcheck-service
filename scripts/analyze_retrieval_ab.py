"""
analyze_retrieval_ab.py
─────────────────────────────────────────────────────────────────────
Análise rigorosa A/B de dois relatórios de `evaluate_retrieval.py`
(ex.: denso vs. híbrido) sobre o MESMO benchmark com campo `split`.

1. Ranking (casos matched): MRR por caso, diferença pareada com IC95%
   por bootstrap pareado e teste de sinal exato (melhoras vs. pioras).
2. Limiar: escolhido SÓ no split `calibration` (maior F1, empate → maior
   limiar), e reportado no split `test` (sem vazamento).

Uso:
    python scripts/analyze_retrieval_ab.py --a reports/retrieval/para_dense.json \
        --b reports/retrieval/para_hybrid.json --eval data/eval/retrieval_benchmark_120_paraphrased.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path


def load_report(path: Path) -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))["cases"]}


def rr(case: dict) -> float:
    return 1.0 / case["rank"] if case.get("rank") else 0.0


def sign_test_p(wins: int, losses: int) -> float:
    """Teste de sinal exato bilateral (empates descartados)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_bootstrap(diffs: list[float], iters: int = 5000, seed: int = 13) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def confusion(cases: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for c in cases:
        predicted = c["best_score"] >= threshold
        if c["expected_status"] == "matched":
            if predicted and c.get("rank") == 1:
                tp += 1
            else:
                fn += 1
                if predicted:
                    fp += 1  # respondeu com o laudo errado em 1º
        else:
            fp += predicted
            tn += not predicted
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def calibrate(cases: list[dict]) -> float:
    grid = [round(x / 100, 2) for x in range(50, 86)]
    return max(grid, key=lambda t: (confusion(cases, t)["f1"], t))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Relatório baseline (ex.: denso)")
    parser.add_argument("--b", type=Path, required=True, help="Relatório candidato (ex.: híbrido)")
    parser.add_argument("--eval", type=Path, required=True, help="Benchmark com campo split")
    parser.add_argument("--out", type=Path, default=Path("reports/retrieval/ab_analysis.json"))
    args = parser.parse_args()

    splits = {}
    for line in args.eval.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            splits[row["id"]] = row.get("split", "test")

    a, b = load_report(args.a), load_report(args.b)
    matched = [cid for cid, c in a.items() if c["expected_status"] == "matched"]
    diffs = [rr(b[cid]) - rr(a[cid]) for cid in matched]
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    mean, low, high = paired_bootstrap(diffs)

    result = {
        "ranking": {
            "n_matched": len(matched),
            "mrr_a": round(sum(rr(a[c]) for c in matched) / len(matched), 4),
            "mrr_b": round(sum(rr(b[c]) for c in matched) / len(matched), 4),
            "recall5_a": round(sum(bool(a[c].get("rank")) for c in matched) / len(matched), 4),
            "recall5_b": round(sum(bool(b[c].get("rank")) for c in matched) / len(matched), 4),
            "mrr_diff_b_minus_a": {"estimate": round(mean, 4), "low": round(low, 4), "high": round(high, 4)},
            "wins_b": wins,
            "losses_b": losses,
            "sign_test_p": round(sign_test_p(wins, losses), 4),
        },
        "threshold": {},
    }
    for name, report in (("a", a), ("b", b)):
        calib = [c for cid, c in report.items() if splits.get(cid) == "calibration"]
        test = [c for cid, c in report.items() if splits.get(cid) == "test"]
        t = calibrate(calib)
        result["threshold"][name] = {
            "chosen_on_calibration": t,
            "calibration": confusion(calib, t),
            "test_at_chosen": confusion(test, t),
            "test_at_0.60": confusion(test, 0.60),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
