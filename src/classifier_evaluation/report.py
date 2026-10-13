"""Machine-readable and human-readable evaluation reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def recommendation(results: list[dict[str, Any]]) -> dict[str, str]:
    evaluated = [result for result in results if result.get("status") == "evaluated"]
    if not evaluated:
        return {
            "decision": "nao_integrar_nenhum_por_enquanto",
            "reason": "não há resultados reais em conjunto externo de teste para sustentar a integração",
        }

    qualified = []
    for result in evaluated:
        metrics = result["metrics"]["calibrated"]
        test_size = result["dataset"]["test_records"]
        if (
            test_size >= 100
            and metrics["f1"] >= 0.70
            and (metrics["roc_auc"] or 0) >= 0.75
            and metrics["ece"] <= 0.10
        ):
            qualified.append(result)
    if not qualified:
        return {
            "decision": "nao_integrar_nenhum_por_enquanto",
            "reason": "nenhum modelo atingiu o gate mínimo documentado (test>=100, F1>=0,70, ROC-AUC>=0,75, ECE<=0,10)",
        }
    if len(qualified) == 1:
        return {
            "decision": f"usar_{qualified[0]['model_name']}",
            "reason": "foi o único modelo que atingiu o gate mínimo no teste externo",
        }

    by_name = {result["model_name"]: result for result in qualified}
    tfidf = by_name.get("tfidf")
    bertimbau = by_name.get("bertimbau")
    if tfidf and bertimbau:
        improvement = bertimbau["metrics"]["calibrated"]["f1"] - tfidf["metrics"]["calibrated"]["f1"]
        if improvement >= 0.03:
            return {
                "decision": "usar_ambos",
                "reason": "BERTimbau agrega >=0,03 de F1; TF-IDF permanece como opção de menor custo",
            }
        return {
            "decision": "usar_tfidf",
            "reason": "ganho de F1 do BERTimbau não compensa sua maior complexidade operacional",
        }
    best = max(qualified, key=lambda result: result["metrics"]["calibrated"]["f1"])
    return {"decision": f"usar_{best['model_name']}", "reason": "melhor F1 entre os modelos aprovados"}


def build_report(
    *,
    artifact_inventory: list[dict[str, object]],
    dataset: dict[str, object],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "offline_classifier_evaluation",
        "artifact_inventory": artifact_inventory,
        "dataset": dataset,
        "results": results,
        "recommendation": recommendation(results),
        "decision_gate": {
            "minimum_test_records": 100,
            "minimum_f1": 0.70,
            "minimum_roc_auc": 0.75,
            "maximum_ece": 0.10,
            "note": "gate inicial conservador; deve ser revisto com responsáveis de produto e risco",
        },
    }


def _format(value: object) -> str:
    return "—" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Avaliação externa de classificadores",
        "",
        "> Este relatório mede padrões textuais. Ele não mede a veracidade factual de uma mensagem.",
        "",
        "## Inventário",
        "",
        "| Modelo | Disponível | Tamanho (bytes) | Versão | Caminho |",
        "|---|---:|---:|---|---|",
    ]
    for artifact in report["artifact_inventory"]:
        lines.append(
            f"| {artifact['name']} | {artifact['available']} | {_format(artifact['size_bytes'])} | "
            f"{_format(artifact['version'])} | {artifact['path'] or 'não configurado'} |"
        )
    lines.extend(["", "## Resultados reais", ""])
    evaluated = [result for result in report["results"] if result.get("status") == "evaluated"]
    if not evaluated:
        lines.append("Nenhum resultado real foi calculado: faltam pesos e/ou dataset externo curado.")
    else:
        lines.extend(
            [
                "| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | ECE | ms/mensagem |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for result in evaluated:
            metrics = result["metrics"]["calibrated"]
            lines.append(
                "| {model} | {accuracy} | {precision} | {recall} | {f1} | {roc} | {pr} | {brier} | {ece} | {latency} |".format(
                    model=result["model_name"],
                    accuracy=_format(metrics["accuracy"]),
                    precision=_format(metrics["precision"]),
                    recall=_format(metrics["recall"]),
                    f1=_format(metrics["f1"]),
                    roc=_format(metrics["roc_auc"]),
                    pr=_format(metrics["pr_auc"]),
                    brier=_format(metrics["brier_score"]),
                    ece=_format(metrics["ece"]),
                    latency=_format(result["latency"]["test_ms_per_message"]),
                )
            )
    decision = report["recommendation"]
    lines.extend(
        [
            "",
            "## Recomendação",
            "",
            f"**{decision['decision']}** — {decision['reason']}.",
            "",
            "Os detalhes completos (calibração, matriz de confusão, curva de confiabilidade, temas, falsos positivos e falsos negativos) estão no JSON correspondente.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "classifier_evaluation.json"
    markdown_path = directory / "classifier_evaluation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    _write_reliability_plot(report, directory / "reliability_curve.png")
    return json_path, markdown_path


def _write_reliability_plot(report: dict[str, Any], path: Path) -> None:
    evaluated = [result for result in report["results"] if result.get("status") == "evaluated"]
    if not evaluated:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="calibração perfeita")
    for result in evaluated:
        curve = result["metrics"]["calibrated"]["reliability_curve"]
        axis.plot(
            [point["mean_probability"] for point in curve],
            [point["fraction_positive"] for point in curve],
            marker="o",
            label=result["model_name"],
        )
    axis.set(xlabel="Probabilidade prevista de falso", ylabel="Fração observada de falso", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
