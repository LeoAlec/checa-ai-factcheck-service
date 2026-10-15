# ADR 001: preservar o layout `src/` no serviço de ML

## Estado

Aceito para o fluxo evidencial e para a avaliação offline.

## Contexto

Este repositório segue o Cookiecutter Data Science e já concentra o runtime em
`src/`, com `app.py` como ponto de composição FastAPI. A constituição também
descreve a arquitetura hexagonal do backend transacional em diretórios
`app/application` e `app/adapters`. Transportar imediatamente todo o serviço de
ML para essa árvore misturaria uma mudança estrutural ampla com a evolução do
contrato evidencial.

## Decisão

O serviço de ML mantém o layout existente em `src/`. As rotas HTTP permanecem
em `app.py` e `src/api/`; contratos públicos ficam em `src/models/`; recuperação,
segurança e integrações de inferência permanecem em módulos explícitos de
`src/rag/` e `src/safety/`. Novas integrações externas devem continuar atrás de
interfaces pequenas e substituíveis nos testes.

`POST /api/debunk` preserva o contrato legado. O contrato evidencial detalhado
é exposto em `POST /api/v2/debunk`, enquanto `POST /api/v1/verify` continua sendo
o contrato compacto entre o serviço de ML e o backend.

## Consequências

- O PR não realiza uma migração de diretórios que afetaria imports, imagens e
  scripts de dados sem ganho funcional imediato.
- A separação de responsabilidades de `FactCheckService` deve evoluir de forma
  incremental, orientada por contratos e testes, sem uma reescrita estrutural.
- Uma futura padronização entre repositórios exigirá um ADR substituto e um PR
  dedicado à migração.
