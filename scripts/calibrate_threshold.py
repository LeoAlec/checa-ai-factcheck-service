"""
calibrate_threshold.py
─────────────────────────────────────────────────────────────────────
Calibra o limiar de similaridade POR provedor/modelo de embedding a partir de
um relatório de `evaluate_retrieval.py` (casos com `best_score` e rank do
laudo esperado) e grava em evals/calibrated_thresholds.json, lido por
`src.config`.

Critério: menor limiar com precisão ≥ `--min-precision` (maximiza cobertura).
Um "acerto" é: o melhor resultado é o laudo esperado. Casos esperados como
`abstained` devem ficar abaixo do limiar (contam como falso positivo se passarem).

Uso:
    python scripts/evaluate_retrieval.py --eval data/eval/retrieval_eval.jsonl
    python scripts/calibrate_threshold.py --min-precision 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import bootstrap_ci, calibrate_threshold, precision_coverage_curve  # noqa: E402

OUT = ROOT / "evals" / "calibrated_thresholds.json"
MIN_CASES = 40  # abaixo disso o limiar não é estatisticamente defensável


def should_answer(row: dict) -> bool:
    """O sistema deve responder se o laudo esperado foi recuperado em 1º lugar."""
    return row["expected_status"] == "matched" and row.get("rank") == 1


def calibrate(report: dict, min_precision: float) -> dict:
    rows = report["cases"]
    scores = [row["best_score"] for row in rows]
    labels = [should_answer(row) for row in rows]
    best = calibrate_threshold(scores, labels, min_precision=min_precision)
    curve = precision_coverage_curve(scores, labels, [round(x / 100, 2) for x in range(30, 91, 5)])
    # IC do limiar por bootstrap sobre casos (estabilidade da escolha).
    import random

    rng = random.Random(13)
    picks = []
    for _ in range(300):
        sample = [rng.randrange(len(rows)) for _ in rows]
        picked = calibrate_threshold([scores[i] for i in sample], [labels[i] for i in sample], min_precision=min_precision)
        if picked["threshold"] is not None:
            picks.append(picked["threshold"])
    return {
        "threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "met_target": best["met_target"],
        "min_precision": min_precision,
        "n_cases": len(rows),
        "threshold_ci95": bootstrap_ci(picks),
        "curve": curve,
        "reliable": len(rows) >= MIN_CASES and best["met_target"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibra limiar por provedor/modelo")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "retrieval" / "retrieval_eval_report.json")
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true", help="grava mesmo com poucos casos/meta não atingida")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    key = f"{report['embedding_provider']}/{report['embedding_model']}"
    result = calibrate(report, args.min_precision)
    print(json.dumps({key: {k: v for k, v in result.items() if k != "curve"}}, indent=2, ensure_ascii=False))

    if not result["reliable"] and not args.force:
        print(f"⚠️  Não gravado: n={result['n_cases']} (mín. {MIN_CASES}) ou precisão-alvo não atingida. Use --force.", file=sys.stderr)
        return 1
    current = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    current[key] = result
    args.out.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Gravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
