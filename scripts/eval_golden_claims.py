"""
eval_golden_claims.py
─────────────────────────────────────────────────────────────────────
Avalia a CONCLUSÃO evidencial (`sustentada`/`contradita`/`inconclusiva`) do
serviço real (claim analyzer + síntese + guardrails + salvaguarda de entidades)
sobre evals/golden_claims.jsonl. A recuperação é substituída pelo laudo do caso
(score fixo acima do limiar): mede-se a decisão, não o retrieval.

Previsão: `matched` → conclusão do modelo; `abstained`/`deep_searched`/`blocked`
→ "inconclusiva" (DeepSearch desligado, como em eval_generation).

Reporta acurácia e macro-F1 com IC 95% (bootstrap), matriz de confusão,
acurácia por fenômeno linguístico e, com `--nli`, o mesmo para o verificador NLI
e a taxa de concordância LLM×NLI.

Uso:
    python scripts/eval_golden_claims.py [--limit 20] [--nli]
Exige OPENAI_API_KEY. Limites mínimos: evals/thresholds.json (`claims_*`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["LANGFUSE_ENABLED"] = "false"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.evaluation.metrics import (  # noqa: E402
    CONCLUSIONS,
    accuracy,
    bootstrap_ci,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    paired_bootstrap_ci,
)

HIGH_SCORE = 0.95  # acima de qualquer limiar: o retrieval não é o objeto do teste


def to_case(item: dict) -> dict:
    return {"id": item["id"], "query": item["claim"], "score": HIGH_SCORE, "evidences": [item["evidence"]]}


def predict_conclusion(result: dict) -> str:
    if result.get("status") == "matched" and result.get("conclusion") in CONCLUSIONS:
        return result["conclusion"]
    return "inconclusiva"


def summarize(rows: list[dict]) -> dict:
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for tag in r["tags"]:
            by_tag[tag].append(r["gold"] == r["pred"])
    return {
        "n": len(rows),
        "accuracy": bootstrap_ci([float(g == p) for g, p in zip(gold, pred)]),
        "macro_f1": paired_bootstrap_ci(gold, pred, macro_f1),
        "confusion": confusion_matrix(gold, pred),
        "accuracy_by_tag": {t: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)} for t, v in sorted(by_tag.items())},
    }


def check_thresholds(summary: dict, th: dict) -> list[str]:
    """Falha pelo LIMITE INFERIOR do IC (não pela estimativa pontual)."""
    failures = []
    for name, key in (("accuracy", "claims_accuracy_min"), ("macro_f1", "claims_macro_f1_min")):
        if key in th and summary[name]["low"] is not None and summary[name]["low"] < th[key]:
            failures.append(f"{name} IC95% inferior = {summary[name]['low']:.3f} < mínimo {th[key]}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval de veredito no golden set de alegações")
    parser.add_argument("--golden", type=Path, default=ROOT / "evals" / "golden_claims.jsonl")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "evals" / "thresholds.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "eval" / "claims.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--nli", action="store_true", help="compara com o verificador NLI auxiliar")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY ausente.", file=sys.stderr)
        return 2
    if args.nli:
        os.environ["NLI_ENABLED"] = "true"

    from eval_generation import build_service

    items = [json.loads(l) for l in args.golden.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]

    rows = []
    for item in items:
        case = to_case(item)
        try:
            pred = predict_conclusion(build_service(case).verify_claim(item["claim"]))
        except Exception as exc:  # noqa: BLE001 - erro do provedor conta como não-resposta
            print(f"⚠️ {item['id']}: {type(exc).__name__}", file=sys.stderr)
            pred = "inconclusiva"
        row = {"id": item["id"], "gold": item["gold_conclusion"], "pred": pred, "tags": item["tags"], "claim": item["claim"]}
        if args.nli:
            from src.rag import nli

            opinion = nli.classify(item["claim"], item["evidence"]["texto"])
            row["nli"] = opinion.conclusion if opinion else None
        rows.append(row)

    summary = summarize(rows)
    if args.nli:
        scored = [r for r in rows if r.get("nli")]
        summary["nli"] = {
            "accuracy": accuracy([r["gold"] for r in scored], [r["nli"] for r in scored]),
            "llm_nli_kappa": cohen_kappa([r["pred"] for r in scored], [r["nli"] for r in scored]),
            "llm_nli_agreement": accuracy([r["pred"] for r in scored], [r["nli"] for r in scored]),
        }
    failures = check_thresholds(summary, json.loads(args.thresholds.read_text()))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"summary": summary, "failures": failures, "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
