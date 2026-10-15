# Data Model

## CalibrationArtifact

- `schema_version`: inteiro positivo.
- `provider` e `model`: identidade do embedding.
- `collection`: índice avaliado.
- `dataset_path` e `dataset_sha256`: proveniência imutável.
- `n_calibration` e `n_test`: tamanhos observados.
- `strategy`: `max_f1` ou `min_precision`.
- `target_precision`: obrigatório somente para `min_precision`.
- `threshold`: valor selecionado.
- `calibration_metrics` e `test_metrics`: métricas separadas.
- `reliable`: resultado dos critérios científicos.
- `promotion_tier`: `pilot`, `production` ou `none`.
- `code_commit` e `generated_at`: lineage.

Transições: `none → pilot → production`. Promoção para produção exige `reliable=true`.

## RetrievalAuditCase

- Identidade, consulta, score, URL retornada e URL esperada.
- Split e limiar da execução.
- `review_category`: vazio até revisão; depois uma categoria permitida.
- `review_decision` e `review_notes`: decisão humana e justificativa.

Categorias: `label_error`, `corpus_gap`, `entity_mismatch`, `time_mismatch`, `topic_only`, `duplicate_or_canonical`, `ambiguous_query`, `embedding_error`, `other`.
