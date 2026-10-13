# Integração da main e DeepSearch na API v1 — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolver o conflito do PR em `checa-ai-ml`, preservando a API v1 e o pipeline da `main`, com DeepSearch apenas como fallback.

**Architecture:** `FactCheckService` compartilha moderação, recuperação, filtro de chunks e fallback da `main`; a v1 mantém sua geração curta e validação de laudos citados. `src/api/v1.py` traduz os quatro resultados para o contrato HTTP, enquanto `/api/debunk` mantém seu corpo de resposta. O container da API recebe as dependências e configurações exigidas pelo código integrado.

**Tech Stack:** Python, FastAPI, Pydantic, pytest, Qdrant, OpenAI, Tavily, Langfuse, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-10-08-ml-main-deepsearch-integration-design.md`.

## Global Constraints

- Trabalhar só em `checa-ai-ml`; não editar `checa-ai`, não fazer push e não reescrever o histórico.
- Base do merge: `origin/main` em `c58a22d` na análise. Atualizar a referência antes de executar e conferir o novo diff se ela avançar.
- `matched` usa exclusivamente 1–2 laudos do Qdrant citados, com URLs HTTP(S); `deep_searched` é fallback neutro com `sources: []`.
- `insufficient_evidence` tem `counter_narrative: null` e `sources: []`; `blocked` tem recusa e `sources: []`.
- `similarity_score` é similaridade do Qdrant; pode ser `null` quando a entrada é bloqueada antes da busca. Nunca tratá-lo como confiança factual.
- Preservar `/api/debunk`, `/api/feedback`, `/health`, shutdown Langfuse, e `/ready`; manter a resposta `matched` dentro do orçamento de 1.000 caracteres do WhatsApp.
- A imagem não indexa dados nem executa DVC. Não registrar alegações, laudos, snippets ou chaves em logs da rota integrada.
- O plano antigo `2026-10-08-ml-backend-integration.md` documenta a implementação original e os trabalhos futuros no outro repositório; este plano cobre apenas a integração do PR atual.

## Review Focus

1. Entrada bloqueada antes da busca deve produzir `blocked` com score nulo e nenhuma chamada ao Qdrant ou Tavily (Task 1).
2. Hit de alto risco de prompt injection não pode aparecer no contexto nem virar fonte `matched` (Task 1).
3. Score baixo ou abstinência do modelo deve acionar DeepSearch; uma falha Tavily deve terminar em `insufficient_evidence`, nunca `matched` (Task 1).
4. URL longa, duplicada, não HTTP(S) ou citação inventada não pode gerar `matched` sem fonte válida nem romper o orçamento da resposta (Task 1).
5. `deep_searched` e `blocked` devem atravessar a rota v1 sem coerção para `matched` ou `insufficient_evidence`, mantendo `sources: []` e nullabilidade do score (Task 2).

---

## Mapa de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `src/fact_check_service.py` | Orquestra pipeline compartilhado e monta resultado v1 sem mudar o corpo de `/api/debunk` |
| `tests/test_rag_pipeline.py` | Regressão da geração v1, fontes, abstinência, bloqueio e fallback |
| `src/api/v1.py`, `tests/test_api_v1.py` | Modelo e transporte HTTP dos quatro vereditos |
| `app.py` | Monta router v1 e preserva rotas, feedback e shutdown da `main` |
| `requirements-api.txt`, `requirements-ci.txt`, `compose.yaml` | Dependências e ambiente da API/CI |
| `.env.example`, `README.md` | Configuração e operação consolidadas após o merge |

## Task 1: Integrar o serviço RAG durante o merge

**Files:** Modify `tests/test_rag_pipeline.py`, `src/fact_check_service.py`, `app.py`, `.env.example`, `README.md`; merge `origin/main`.

**Interfaces:** Keep `FactCheckService.verify_claim(user_message: str, *, session_id: str | None = None) -> dict` for `/api/debunk`; keep `FactCheckService.verify_claim_for_backend(text: str) -> dict` returning `schema_version`, `verdict`, `similarity_score`, `counter_narrative`, `sources`; keep `VerificationGenerationError`.

- [ ] **Step 1: Start the merge.** Refresh `origin/main`; on the clean named branch, run `git merge --no-commit --no-ff origin/main`. Inspect `git status --short` and compare conflicts with the four known paths. If `main` advanced, inspect the new changes before resolving.

- [ ] **Step 2: Restore a runnable baseline.** Resolve `.env.example`, `README.md`, `app.py`, and `src/fact_check_service.py` from both parents. Preserve router inclusion and threadpool startup from this branch; preserve feedback, web route, Langfuse shutdown, prompts, safety and DeepSearch from `main`. In the service, initially keep both existing methods without adding integration behavior. Run `.venv/bin/python -m compileall -q app.py src`; expect exit 0.

- [ ] **Step 3: Prepare the test environment.** Run `.venv/bin/python -m pip install -r requirements-ci.txt pytest-asyncio`; the current venv lacks `tavily`. Verify `.venv/bin/python -c 'import tavily, langfuse, pytest_asyncio'` exits 0.

- [ ] **Step 4: Write failing integration tests.** In `tests/test_rag_pipeline.py`, add `test_v1_blocks_input_before_retrieval`, `test_v1_excludes_high_risk_chunk`, `test_v1_uses_deep_search_below_threshold`, `test_v1_uses_deep_search_after_model_abstention`, `test_v1_blocks_unsafe_generated_output`, `test_v1_does_not_log_claim_or_web_snippets`, and `test_v1_generation_does_not_use_debunk_prompt_version`. Stub network boundaries. Assert blocked input yields `result["verdict"] == "blocked"`, `result["similarity_score"] is None`, `result["sources"] == []` and no retrieval call; high-risk URL is absent from matched sources/context; both fallback triggers yield `result["verdict"] == "deep_searched"` and `result["sources"] == []`; unsafe output is blocked; captured logs exclude the claim and snippets; generation metadata uses `verify-v1`. Keep existing insufficient-evidence, cited-source and length tests. Add a test that an invented URL in a `matched` narrative is rejected or sanitized before response.

- [ ] **Step 5: Verify red.** Run `.venv/bin/python -m pytest tests/test_rag_pipeline.py -q -k 'v1_blocks_input or v1_excludes_high_risk or v1_uses_deep_search or v1_blocks_unsafe_generated or v1_does_not_log or v1_generation_does_not_use'`. Expect assertion failures because the unintegrated v1 method bypasses the `main` guardrails and fallback; import/setup errors are not an acceptable red result.

- [ ] **Step 6: Implement the service integration.** In `src/fact_check_service.py`, make `verify_claim_for_backend` use the same input guardrails, filtered `_retrieve`, abstention decision, DeepSearch fallback, output validation and tracing path as the web flow while retaining its compact `matched` generation and citation/URL/length checks. Use `parse_answer` to recognize `laudos_tratam_alegacao: false`; identify the v1 generation as `verify-v1` rather than the Langfuse `debunk` prompt/version. Map fallback `deep_searched` to v1 with `sources: []`; map no web result to `insufficient_evidence`. Remove raw query and web-result logging on the shared fallback path.

- [ ] **Step 7: Verify green.** Run `.venv/bin/python -m pytest tests/test_rag_pipeline.py tests/test_guardrails.py tests/test_observability.py tests/test_prompt_contract.py -q`; expect zero failures. Run `git diff --check` and `git diff --name-only --diff-filter=U`; expect no whitespace errors or unresolved paths. Keep the merge open through Tasks 2–3; Git cannot make intermediate commits inside a merge.

## Task 2: Atualizar o contrato HTTP v1

**Files:** Modify `src/api/v1.py`, `tests/test_api_v1.py`; adjust `app.py` only if routing requires it.

**Interfaces:** `VerifyResponse.verdict: Literal["matched", "deep_searched", "insufficient_evidence", "blocked"]`; `similarity_score: float | None`; `sources: list[VerificationSource]`. `POST /api/v1/verify` still accepts nonblank `text` of 1–10.000 characters; `GET /ready` remains the readiness check.

- [ ] **Step 1: Write failing API tests.** In `tests/test_api_v1.py`, extend `test_versioned_results_serialize_exactly` or add parameterized cases for all four verdicts, plus `test_openapi_verification_schema_has_four_verdicts_and_nullable_score`. Assert `response.json()["verdict"] == "deep_searched"`, `response.json()["sources"] == []`, and a blocked response has `response.json()["similarity_score"] is None`; assert OpenAPI's verdict enum is exactly `{matched, deep_searched, insufficient_evidence, blocked}` and score is nullable. Retain tests for 422, generation 5xx, readiness 503, slow request/health concurrency and the current `/api/debunk` response shape.

- [ ] **Step 2: Verify red.** Run `.venv/bin/python -m pytest tests/test_api_v1.py -q -k 'versioned_results or openapi_verification_schema'`; expect validation or schema assertion failures for the two new verdicts or nullable score, with existing cases still valid.

- [ ] **Step 3: Implement schema and mapping.** Update `VerifyResponse` in `src/api/v1.py` to those exact enum/nullability values; keep `VerificationSource` as `title/domain/url` with no Tavily source field. Preserve `VerificationGenerationError -> 502` and unexpected/unavailable -> 503 without a verdict body. Ensure `app.py` includes the v1 router alongside the `main` routes.

- [ ] **Step 4: Verify green.** Run `.venv/bin/python -m pytest tests/test_api_v1.py tests/test_rag_pipeline.py -q`; expect zero failures. Resolve any interaction failures using their failing test as the reproduction.

- [ ] **Step 5: Record the verified contract.** Run `.venv/bin/python -m pytest tests/test_api_v1.py -q -k 'openapi_verification_schema'`; expect pass. Keep the merge open for the runtime gate in Task 3.

## Task 3: Fechar dependências, configuração e documentação

**Files:** Modify `requirements-api.txt`, `requirements-ci.txt`, `compose.yaml`, `.env.example`, `README.md`; add a narrow startup/import test only if build/import checks expose a gap.

**Interfaces:** `ml-api` keeps `QDRANT_URL=http://qdrant:6333`; Compose passes `TAVILY_API_KEY` and Langfuse settings when configured. Missing Tavily key leaves fallback without web results and does not make `/ready` fail; readiness still checks OpenAI key, collection and vector dimension.

- [ ] **Step 1: Show the missing Compose environment.** Run `docker compose config -q`, then `docker compose config --format json | python -c 'import json,sys; e=json.load(sys.stdin)["services"]["ml-api"]["environment"]; assert "TAVILY_API_KEY" in e and "LANGFUSE_PUBLIC_KEY" in e'`; expect the assertion to fail before Step 3.

- [ ] **Step 2: Show the missing image imports.** If Docker is available, run `docker compose build ml-api` and `docker compose run --rm --no-deps ml-api python -c 'import app'`; expect the import to fail because the current API image lacks Tavily/Langfuse. If Docker is unavailable, record the exact blocker instead of claiming a build result.

- [ ] **Step 3: Close the runtime dependencies.** Add the direct runtime imports of the integrated API to `requirements-api.txt`, including `tavily-python` and `langfuse>=4,<5`; add `pytest-asyncio` to `requirements-ci.txt` because `tests/test_api_v1.py` imports it.

- [ ] **Step 4: Pass the runtime environment.** Set `TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LANGFUSE_ENABLED`, `LANGFUSE_CAPTURE_CONTENT` and `PROMPT_LABEL` in `compose.yaml` from environment variables, without hard-coded secrets.

- [ ] **Step 5: Consolidate docs.** Keep both branches' README sections and `.env.example` entries. Document that v1 `matched` returns cited corpus sources, `deep_searched` returns a cautious summary with no structured sources, and the sibling backend must handle the enlarged enum before use. Document startup/readiness with an existing indexed collection; do not add ingestion.

- [ ] **Step 6: Verify the image and Compose.** Run `docker compose config -q` and the environment assertion from Step 1; run `docker compose build ml-api` and `docker compose run --rm --no-deps ml-api python -c 'import app'` if Docker works. Expect exit 0, or report the exact Docker blocker.

- [ ] **Step 7: Verify the full tree.** Run `.venv/bin/python -m pytest tests -q` and `.venv/bin/python -m compileall -q app.py src`; expect exit 0. Check `git diff --check`, `git status --short`, and the captured-log regression test from Task 1.

- [ ] **Step 8: Complete merge commit.** Stage all resolved and added files, run `git diff --cached --check`, and verify `git diff --name-only --diff-filter=U` is empty. Commit with `git commit -m "merge: integrate main with versioned verification API"`, without rewriting branch history. If the final test gate is red, fix it before committing.

## Final verification and handoff

- Re-run the full test suite and `docker compose config -q` on the final tree; verify both `/api/debunk` and `/api/v1/verify` routes exist and `/ready` retains its readiness conditions.
- If Docker and an indexed Qdrant collection are available, smoke test `matched`, `deep_searched`, `insufficient_evidence` and `blocked` without treating these contract checks as a measure of factual quality.
- Compare final tree to `origin/main` and the pre-merge branch; inspect `git status` for unmerged paths and conflict markers. Report test results, environment limits and the local merge commit. Do not push.
