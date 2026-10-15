# Quickstart de validação

1. Execute os testes unitários novos:

   `python -m pytest -q tests/test_calibrate_threshold.py tests/test_retrieval_audit.py tests/test_evidence_consistency.py`

2. Calibre com o split de calibração e avalie no teste:

   `python scripts/calibrate_threshold.py --strategy max-f1 --promotion-tier pilot`

3. Confirme no JSON gerado que o dataset possui hash, splits separados e métricas separadas.

4. Gere a planilha dos falsos positivos:

   `python scripts/audit_retrieval_errors.py --report reports/retrieval/retrieval_eval_report.json --output reports/retrieval/false_positive_review.csv`

5. Em configuração de produção, confirme que um artefato de piloto não altera o default. Em configuração de piloto, confirme que o opt-in carrega o valor.

## Resultado local em 2026-10-14

- 35 testes diretamente relacionados passaram.
- Na validação ampliada, 98 testes passaram; dois testes não executaram por ausência de `openai` no ambiente disponível e três fixtures de API não executaram por ausência de `fastapi`.
- O executável Python global está inconsistente; a validação utilizou `.venv-classifier-evaluation` com `--basetemp` dentro do workspace.
- O CSV gerado contém os dez falsos positivos do split de teste no limiar 0,65.
