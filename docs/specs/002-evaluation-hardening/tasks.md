# Tasks: Endurecimento da avaliação de retrieval

## Phase 1: Setup

- [x] T001 Registrar especificação e design em `docs/specs/002-evaluation-hardening/`

## Phase 2: Foundational

- [x] T002 Criar contratos tipados de calibração em `src/evaluation/calibration.py`

## Phase 3: User Story 1 - Promover limiares com segurança (P1)

**Independent Test**: seleção não acessa teste e runtime respeita confiabilidade/tier.

- [x] T003 [US1] Criar testes de estratégias, provenance e promoção em `tests/test_calibrate_threshold.py`
- [x] T004 [US1] Implementar estratégias e artefato em `src/evaluation/calibration.py`
- [x] T005 [US1] Adaptar CLI em `scripts/calibrate_threshold.py`
- [x] T006 [US1] Aplicar política segura no runtime em `src/config.py`
- [x] T007 [US1] Corrigir e documentar o artefato atual em `evals/calibrated_thresholds.json`

## Phase 4: User Story 2 - Auditar erros de retrieval (P2)

**Independent Test**: relatório conhecido exporta exatamente seus falsos positivos.

- [x] T008 [US2] Criar testes de exportação/validação em `tests/test_retrieval_audit.py`
- [x] T009 [US2] Implementar domínio de auditoria em `src/evaluation/audit.py`
- [x] T010 [US2] Criar CLI em `scripts/audit_retrieval_errors.py`

## Phase 5: User Story 3 - Rastrear bloqueios e consistência (P2)

**Independent Test**: conclusão factual sem fonte falha de forma segura e bloqueios emitem log.

- [x] T011 [US3] Criar testes de consistência e logging em `tests/test_evidence_consistency.py`
- [x] T012 [US3] Implementar invariante determinístico em `src/rag/guardrails.py`
- [x] T013 [US3] Padronizar logs de bloqueio em `src/fact_check_service.py`

## Phase 6: Polish

- [x] T014 Atualizar metodologia e operação em `docs/RETRIEVAL_BENCHMARK.md` e `README.md`
- [x] T015 Executar testes relevantes e registrar limitações do ambiente em `docs/specs/002-evaluation-hardening/quickstart.md`

## Dependencies

- T002 bloqueia T004–T007.
- T003 precede T004–T007.
- T008 precede T009–T010.
- T011 precede T012–T013.
- T014–T015 dependem das três histórias.

## Implementation Strategy

MVP: concluir US1 antes de auditoria e guardrails. Cada história possui teste independente e não altera o contrato HTTP.
