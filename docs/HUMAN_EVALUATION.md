# Avaliação humana do fluxo evidencial

Esta avaliação mede o fluxo completo da API local, sem modelos classificatórios
e sem dependência ou alteração de `checa-ai-backend`. Cada caso captura a resposta
de `/api/debunk`; revisores humanos avaliam a alegação extraída, fontes retornadas,
conclusão, abstenção e URLs citadas.

## Protocolo

1. Para os casos locais favoráveis, gere rascunhos a partir da coleção existente.
   Esta etapa apenas lê pontos; não indexa, recria ou altera a coleção:

   ```powershell
   python scripts/build_human_evaluation_candidates.py `
     --qdrant-path "C:\Users\leozi\OneDrive\Área de Trabalho\projetos\checa-ai-ml\data\qdrant_db" `
     --allow-local-reference-index `
     --collection fact_checks_pt_v2_openai `
     --limit 10 `
     --output data/evaluation/candidatos_locais.jsonl
   ```

   Abra `data/evaluation/candidatos_locais.jsonl`, mantenha quatro casos com
   alegação factual clara e copie-os para um arquivo local não versionado, por
   exemplo `data/evaluation/minha_amostra.jsonl`. Em seguida acrescente casos
   contraditos, vagos, URLs isoladas, evidência semanticamente parecida mas sobre
   outro fato e falhas de fonte, até alcançar pelo menos 20 casos.
2. Com o serviço e os serviços reais configurados, capture as respostas:

   ```powershell
   python scripts/run_human_evaluation.py `
     --cases data/evaluation/minha_amostra.jsonl `
     --output reports/human-evaluation/responses.jsonl
   ```

3. Copie `data/evaluation/human_review_template.csv`, registre uma adjudicação
   `approved` por caso e use ao menos um `reviewer_id`. Cada dimensão
   recebe `pass`, `fail` ou `na`; use `na` somente se ela realmente não se aplicar.
4. Gere o relatório reproduzível:

   ```powershell
   python scripts/evaluate_human_review.py `
     --responses reports/human-evaluation/responses.jsonl `
     --reviews data/evaluation/minha_revisao.csv `
     --output-dir reports/human-evaluation
   ```

## Critérios de aprovação

O relatório permite integração com o backend somente se houver 20 ou mais casos,
um ou mais revisores, exatamente uma adjudicação aprovada por caso e todos os
limiares abaixo:

| Dimensão | Limiar |
| --- | --- |
| Extração da alegação | 90% |
| Pertinência das fontes | 90% |
| Conclusão | 90% |
| Abstenção | 95% |
| Grounding das citações | 100% |

Qualquer amostra, adjudicação ou métrica insuficiente gera `not_ready` e código
de saída `2`. O responsável pelo serviço pode aprovar conscientemente uma execução
com exceções registradas, sem esconder as métricas, usando:

```powershell
python scripts/evaluate_human_review.py `
  --responses reports/human-evaluation/responses.jsonl `
  --reviews data/evaluation/minha_revisao.csv `
  --output-dir reports/human-evaluation `
  --approve-with-exceptions-by leo
```

Esse modo retorna `approved_by_owner_with_exceptions`, registra quem aprovou e
mantém todas as exceções no relatório. Os arquivos em `reports/` são resultados
locais e não devem ser versionados.
