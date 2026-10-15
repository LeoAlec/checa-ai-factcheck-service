# Implementation Plan: Endurecimento da avaliação de retrieval

**Branch**: `feat/nlp-review-retrieval-eval` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

## Summary

Separar seleção, validação e promoção de limiares; tornar os artefatos reproduzíveis; exportar falsos positivos para revisão humana; e fechar lacunas determinísticas de grounding e auditoria sem ativar novos modelos.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: biblioteca padrão, Pydantic existente, pytest

**Storage**: JSON para artefatos e CSV para revisão humana

**Testing**: pytest, testes unitários sem rede

**Target Platform**: serviço Linux/contêiner e scripts multiplataforma

**Project Type**: serviço web com ferramentas offline de avaliação

**Performance Goals**: carregamento de configuração em tempo constante; auditoria linear no número de casos

**Constraints**: não alterar contrato HTTP; não depender de modelos ou serviços externos; preservar compatibilidade com artefatos legados de forma segura

**Scale/Scope**: benchmark atual de 120 casos, com suporte a conjuntos maiores

## Constitution Check

- SDD: atendido por spec, plano e tarefas desta feature.
- Test-first: testes precedem as mudanças de comportamento.
- Segurança: nenhum conteúdo de usuário é incluído nos novos logs.
- Python 3.11+ e type hints: obrigatórios nos novos contratos.
- Arquitetura: lógica de avaliação permanece isolada de adaptadores externos.

Reavaliação pós-design: nenhum gate constitucional adicional foi violado.

## Project Structure

```text
docs/specs/002-evaluation-hardening/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/calibration-artifact.schema.json
└── tasks.md

src/evaluation/
├── calibration.py
└── audit.py

scripts/
├── calibrate_threshold.py
└── audit_retrieval_errors.py

tests/
├── test_calibrate_threshold.py
├── test_retrieval_audit.py
└── test_evidence_consistency.py
```

**Structure Decision**: manter ferramentas científicas em `src/evaluation` e CLIs finos em `scripts`, sem ampliar o acoplamento de `FactCheckService`.
