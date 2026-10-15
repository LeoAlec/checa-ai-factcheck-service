# Revisão de PLN: alterações e próximos passos

Branch: `feat/nlp-review-retrieval-eval`. Esta revisão partiu de uma avaliação do projeto sob a ótica de PLN. Aponta o que foi implementado, o que ficou **pendente ou não validado** e o que fazer a seguir.

## O que foi alterado

### Retrieval
| Mudança | Arquivo | Observação |
|---|---|---|
| Fusão híbrida BM25 + denso/cluster por RRF | `src/rag/hybrid.py` | Ligada por padrão (`HYBRID_ENABLED`). Não altera o `score` usado no limiar, só a ordem de seleção. |
| Reranker cross-encoder opcional | `src/rag/reranker.py` | Desligado (`RERANKER_ENABLED=false`). Cai para a ordem anterior se falhar. |
| Limiar de similaridade por provedor/modelo | `src/config.py`, `scripts/calibrate_threshold.py` | Lê `evals/calibrated_thresholds.json` se existir. Env `SIMILARITY_THRESHOLD` tem prioridade. |

### Entidades e NLI
| Mudança | Arquivo | Observação |
|---|---|---|
| Extração de entidades (spaCy opcional + heurística) | `src/rag/entities.py` | Detecta siglas, aliases (STF↔Supremo…), fronteira de palavra. Em CAIXA ALTA não infere entidades. |
| NLI auxiliar opcional | `src/rag/nli.py` | Desligado. Só registra concordância com o LLM; com `NLI_ABSTAIN_ON_DISAGREEMENT=true` força `inconclusiva`. |
| Integração no serviço | `src/fact_check_service.py` | `_specific_named_entities` e `_contradiction_has_entity_anchor` agora delegam a `entities.py`. |

### Avaliação
| Mudança | Arquivo |
|---|---|
| Métricas: recall@k, MRR, nDCG, macro-F1, kappa, curva precisão×cobertura, bootstrap (IC95%) | `src/evaluation/metrics.py` |
| `evaluate_retrieval.py` usa o pipeline do serviço e reporta ranking com IC | `scripts/evaluate_retrieval.py` |
| Gerador do golden set (100 alegações, gerador + juiz LLM cego) | `scripts/build_golden_claims.py` |
| Avaliador de veredito no golden set (IC por bootstrap, por fenômeno, comparação com NLI) | `scripts/eval_golden_claims.py` |
| Limiares `claims_*` (avaliados pelo limite inferior do IC) | `evals/thresholds.json` |
| Testes (métricas, híbrido, entidades, gerador) | `tests/test_eval_metrics.py`, `tests/test_hybrid_entities.py`, `tests/test_build_golden_claims.py` |

### Outras
- `LLM_TEMPERATURE` padrão passou de 0.2 para **0.0** (reprodutibilidade em tarefa de classificação). Pode mudar respostas em relação à versão anterior.
- `.env.example`, `README.md` e `requirements-nlp-advanced.txt` (spaCy, transformers) atualizados.

Suíte local: 203 testes passando (sem `tests/test_safety_e2e.py`, que exige serviços reais).

## Estado validado em 2026-10-14

1. O responsável do projeto confirmou os rótulos do golden set de 100 alegações. Vinte decisões individuais e o protocolo de concordância global estão registrados em `evals/golden_claims_human_decisions.json`; revisões LLM permanecem artefatos auxiliares, não evidência humana.
2. O prompt v4 foi avaliado em held-out independente de 60 casos: acurácia 0,883 (IC95% 0,800–0,950) e macro-F1 0,888 (IC95% 0,798–0,955).
3. Uma amostra estratificada de 20 itens desse held-out recebeu revisão humana completa, com concordância de 20/20 com os rótulos de referência. O manifesto congelado está em `evals/heldout_claims_v1_manifest.json`.
4. Retrieval foi medido em 120 casos reais/parafraseados. O híbrido não apresentou ganho estatisticamente demonstrável e permanece desligado; o limiar 0,65 foi selecionado apenas em calibração e melhorou o teste cego.
5. O NLI foi executado em L4 e ficou abaixo do LLM, inclusive na estratégia de abstenção por discordância. Ele permanece desligado.
6. O reranker BGE v2-m3 foi medido em L4 sobre o benchmark offline e piorou o MRR de 0,933 para 0,889 (delta -0,044; IC95% -0,097 a +0,006; teste de sinal p=0,180). Ele permanece desligado; ver `reports/retrieval/reranker_bge_v2_m3_summary.md`.
7. A heurística de entidades ainda tem cobertura limitada para nomes próprios de uma palavra; o backend spaCy não foi avaliado end-to-end.

## Próximos passos (em ordem)

1. Quando houver credenciais, publicar `evidence-flow-v4` no label de staging do Langfuse e executar o canário documentado em `docs/LANGFUSE_PROMPT_STATUS.md`; produção continua bloqueada até esse passo.
2. Executar o smoke real de API descrito em `docs/LIVE_E2E.md` com coleção dedicada, preservando o índice real.
3. Construir um novo conjunto de desenvolvimento para um eventual prompt v5, orientado pela taxonomia dos sete erros do held-out. Não reutilizar o held-out v1 para ajuste.
4. Avaliar spaCy separadamente por cobertura de entidades, qualidade e latência. NLI, híbrido e reranker permanecem desligados até nova evidência.

O FakeWhatsApp.BR local é uma fonte real e declaradamente anonimizada, mas seus rótulos medem padrão textual de desinformação, não a conclusão evidencial do RAG. Ele deve continuar separado do benchmark alegação→laudo.
