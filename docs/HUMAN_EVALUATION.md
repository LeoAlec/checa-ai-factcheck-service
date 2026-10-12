# Avaliação humana do fluxo evidencial

Esta avaliação mede o fluxo completo da API local, sem modelos classificatórios
e sem dependência ou alteração de `checa-ai-backend`. Cada caso captura a resposta
de `/api/debunk`; revisores humanos avaliam a alegação extraída, fontes retornadas,
conclusão, abstenção e URLs citadas.

## Protocolo

1. Copie `data/evaluation/human_review_cases.template.jsonl` para um arquivo local
   não versionado e substitua os exemplos por pelo menos 20 casos representativos.
   Inclua alegações sustentadas, contraditas, vagas, links isolados, evidência
   semanticamente parecida mas sobre outro fato e falhas de fonte.
2. Com o serviço e os serviços reais configurados, capture as respostas:

   ```powershell
   python scripts/run_human_evaluation.py `
     --cases data/evaluation/minha_amostra.jsonl `
     --output reports/human-evaluation/responses.jsonl
   ```

3. Copie `data/evaluation/human_review_template.csv`, registre uma adjudicação
   `approved` por caso e use ao menos dois `reviewer_id` distintos. Cada dimensão
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
dois ou mais revisores, exatamente uma adjudicação aprovada por caso e todos os
limiares abaixo:

| Dimensão | Limiar |
| --- | --- |
| Extração da alegação | 90% |
| Pertinência das fontes | 90% |
| Conclusão | 90% |
| Abstenção | 95% |
| Grounding das citações | 100% |

Qualquer amostra, adjudicação ou métrica insuficiente gera `not_ready` e código
de saída `2`. Os arquivos em `reports/` são resultados locais e não devem ser
versionados.
