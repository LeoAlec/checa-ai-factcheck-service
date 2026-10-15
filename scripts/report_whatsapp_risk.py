"""Create a compact model-card style report for WhatsApp risk experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/whatsapp_risk.manifest.json"))
    parser.add_argument("--metrics", type=Path, default=Path("models/whatsapp_risk_baselines/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/whatsapp-risk/baseline-report.md"))
    args = parser.parse_args()

    metadata = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = metadata["manifest"]
    results = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = []
    for name, model in results["models"].items():
        default = model["splits"]["test"]["default_threshold"]
        selected = model["splits"]["test"]["validation_selected_threshold"]
        rows.append(
            f"| {name} | {default['f1_macro']:.3f} | {default['f1_misinformation']:.3f} | "
            f"{default['roc_auc']:.3f} | {default['pr_auc']:.3f} | {default['ece']:.3f} | "
            f"{selected['threshold']:.3f} | {selected['f1_macro']:.3f} |"
        )
    content = f"""# Relatório — risco textual em mensagens de WhatsApp

## Escopo

Este classificador responde **“a mensagem se parece com mensagens rotuladas como desinformação?”**.
Ele não estima a verdade factual, não substitui busca de evidências e não deve bloquear conteúdo sozinho.

## Dados e protocolo

- Fonte: FakeWhatsApp.BR (`{manifest['input_sha256']}`)
- Linhas brutas: {manifest['raw_rows']:,}
- Linhas manualmente rotuladas 0/1: {manifest['labeled_rows']:,}
- Textos rotulados únicos após normalização: {manifest['unique_labeled_texts']:,}
- Exemplos exportados: {manifest['exported_rows']:,}
- Divisão temporal: treino até agosto/2018, validação setembro/2018, teste outubro/2018 em diante
- Grupos repetidos entre períodos excluídos: {manifest['cross_period_groups_excluded']:,}
- Rótulo `-1`: não rotulado, sempre excluído

## Teste temporal intocado

| Modelo | Macro-F1 @0,5 | F1 desinformação | ROC-AUC | PR-AUC | ECE | limiar da validação | Macro-F1 no limiar |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

O limiar foi escolhido exclusivamente na validação. Como o ajuste aumentou recall mas reduziu macro-F1 no
teste, o limiar 0,5 é a referência mais prudente até existir validação prospectiva no tráfego do produto.

## Decisão recomendada

1. Manter os modelos TF-IDF como baseline auditável e fallback barato.
2. Comparar o BERTimbau no mesmo teste temporal antes de promover qualquer artefato.
3. Exibir o resultado como `risco_textual_whatsapp`, separado de `risco_editorial_artigo` e do veredito factual.
4. Coletar exemplos reais do produto, com consentimento e anonimização, para medir drift e calibrar o limiar.
5. Não treinar com as 282 mil linhas: cerca de 261 mil têm rótulo `-1` (desconhecido), não negativo.

## Limitações

- Os rótulos foram propagados a ocorrências idênticas e cobrem um período curto de 2018.
- A distribuição de mensagens virais do corpus pode diferir do uso atual do produto.
- Deduplicação exata normalizada reduz vazamento óbvio, mas paráfrases próximas ainda podem atravessar splits.
- Probabilidade do modelo é escore de semelhança de domínio, não probabilidade de falsidade.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
