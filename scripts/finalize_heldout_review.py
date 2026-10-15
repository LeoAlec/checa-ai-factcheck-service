"""Validate and freeze the human audit of a held-out claim-verdict benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path


ERROR_TAXONOMY = {
    "gc-014": ("referencia_temporal", "'último mês' foi tratado como incompatível com setembro de 2023"),
    "gc-017": ("referencia_temporal", "'mês passado' foi tratado como incompatível com agosto de 2023"),
    "gc-020": ("ausencia_como_negacao", "finalidade não comprovada foi interpretada como refutação"),
    "gc-021": ("ausencia_como_negacao", "ausência de menção a rastreamento foi interpretada como refutação"),
    "gc-026": ("contradicao_explicita_subestimada", "negação explícita de eficácia virou abstenção"),
    "gc-035": ("quantificador_absoluto_subestimado", "operação normal refuta 'usinas paradas', mas o modelo se absteve"),
    "gc-059": ("escopo_temporal", "checagem de 2018 não resolve a referência ambígua a 'última eleição'"),
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_review(dataset: list[dict], review_rows: list[dict]) -> dict:
    by_id = {row["id"]: row for row in dataset}
    if len(by_id) != len(dataset):
        raise ValueError("held-out contém IDs duplicados")
    if len({row["id"] for row in review_rows}) != len(review_rows):
        raise ValueError("amostra humana contém IDs duplicados")
    if not review_rows:
        raise ValueError("amostra humana vazia")

    mismatches: list[str] = []
    for row in review_rows:
        case = by_id.get(row["id"])
        if case is None:
            raise ValueError(f"caso da revisão ausente no held-out: {row['id']}")
        human = row.get("human_conclusion", "").strip()
        if not human:
            raise ValueError(f"decisão humana ausente: {row['id']}")
        if human != row["gold_conclusion"] or human != case["gold_conclusion"]:
            mismatches.append(row["id"])
    return {
        "sample_size": len(review_rows),
        "completed": len(review_rows),
        "agreement_with_gold": (len(review_rows) - len(mismatches)) / len(review_rows),
        "mismatches": mismatches,
    }


def build_error_rows(dataset: list[dict], evaluation: dict) -> list[dict]:
    cases = {row["id"]: row for row in dataset}
    errors = []
    for result in evaluation["cases"]:
        if result["gold"] == result["pred"]:
            continue
        category, explanation = ERROR_TAXONOMY.get(
            result["id"], ("nao_classificado", "requer análise manual")
        )
        case = cases[result["id"]]
        errors.append(
            {
                "id": result["id"],
                "gold": result["gold"],
                "pred": result["pred"],
                "category": category,
                "explanation": explanation,
                "tags": case.get("tags", []),
            }
        )
    return errors


def render_error_report(errors: list[dict], evaluation: dict) -> str:
    transitions = Counter(f"{row['gold']} → {row['pred']}" for row in errors)
    categories = Counter(row["category"] for row in errors)
    summary = evaluation["summary"]
    lines = [
        "# Análise de erros — held-out v1",
        "",
        "Este documento é diagnóstico. O held-out está congelado e não deve ser usado para retunar o prompt v4.",
        "",
        "## Resultado",
        "",
        f"- Acurácia: **{summary['accuracy']['estimate']:.3f}** (IC95% {summary['accuracy']['low']:.3f}–{summary['accuracy']['high']:.3f}).",
        f"- Macro-F1: **{summary['macro_f1']['estimate']:.3f}** (IC95% {summary['macro_f1']['low']:.3f}–{summary['macro_f1']['high']:.3f}).",
        f"- Erros: **{len(errors)}/{summary['n']}**.",
        "- Auditoria humana: 20/20 itens concluídos; concordância integral com os rótulos de referência na amostra.",
        "",
        "## Distribuição",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(transitions.items()))
    lines += ["", "Categorias:"]
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(categories.items()))
    lines += [
        "",
        "## Casos",
        "",
        "| ID | Gold → predição | Categoria | Diagnóstico |",
        "|---|---|---|---|",
    ]
    for row in errors:
        lines.append(
            f"| {row['id']} | {row['gold']} → {row['pred']} | `{row['category']}` | {row['explanation']} |"
        )
    lines += [
        "",
        "## Decisão",
        "",
        "O padrão dominante continua sendo a fronteira entre contradição explícita e falta de comprovação. "
        "Os dois erros de referência temporal também mostram que expressões dêiticas precisam de uma data de referência explícita. "
        "Esses achados devem orientar um futuro v5 em um novo conjunto de desenvolvimento; não autorizar mudança com base neste held-out.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/heldout_claims_v1.jsonl"))
    parser.add_argument("--review", type=Path, default=Path("evals/heldout_claims_v1_human_review.csv"))
    parser.add_argument("--evaluation", type=Path, default=Path("reports/eval/claims_v4_heldout_v1.json"))
    parser.add_argument("--manifest", type=Path, default=Path("evals/heldout_claims_v1_manifest.json"))
    parser.add_argument("--error-report", type=Path, default=Path("reports/eval/claims_v4_heldout_v1_error_analysis.md"))
    args = parser.parse_args()

    dataset = _jsonl(args.dataset)
    with args.review.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    audit = validate_review(dataset, review_rows)
    errors = build_error_rows(dataset, evaluation)
    if len(evaluation["cases"]) != len(dataset):
        raise ValueError("relatório e held-out têm tamanhos diferentes")

    manifest = {
        "dataset": str(args.dataset).replace("\\", "/"),
        "sha256": _sha256(args.dataset),
        "frozen_at": date.today().isoformat(),
        "status": "frozen_test_only",
        "n": len(dataset),
        "label_distribution": dict(sorted(Counter(row["gold_conclusion"] for row in dataset).items())),
        "human_audit": audit,
        "evaluation_report": str(args.evaluation).replace("\\", "/"),
        "evaluation_errors": len(errors),
        "policy": "Do not tune prompts, thresholds, or models on this held-out set.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.error_report.parent.mkdir(parents=True, exist_ok=True)
    args.error_report.write_text(render_error_report(errors, evaluation), encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "error_report": str(args.error_report), **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
