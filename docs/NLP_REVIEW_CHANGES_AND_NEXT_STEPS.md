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

## O que NÃO foi feito ou validado

1. **O golden set de 100 alegações ainda não foi gerado/versionado.** A execução de `scripts/build_golden_claims.py` estava em andamento quando esta branch foi criada. O arquivo `evals/golden_claims.jsonl` e o relatório não estão neste commit. Rode o script e revise o relatório (concordância gerador×juiz, kappa, distribuição por rótulo).
2. **O golden set é sintético e verificado por LLM, não por humanos.** Mede consistência alegação↔laudo, não verdade no mundo, e herda vieses do gerador e do juiz.
3. **Nenhuma métrica de retrieval foi medida.** Não há base com laudos reais anotados nem Qdrant executado nesta revisão. O impacto do híbrido, o peso `HYBRID_LEXICAL_WEIGHT=1.2` e o limiar calibrado são **hipóteses a validar**.
4. **Reranker, spaCy e NLI não foram executados** (modelos não baixados). Só a lógica de fallback foi testada.
5. **Os repositórios `checa-ai` e `checa-ai-data` não foram acessados** (fora do workspace). O `retrieval_report.json` aberto não foi analisado.
6. A heurística de entidades só detecta nomes com 2 ou mais palavras capitalizadas e siglas; nomes de uma palavra só (ex.: "Anvisa") dependem do spaCy.

## Próximos passos (em ordem)

1. Gerar e inspecionar o golden set: `python scripts/build_golden_claims.py`. Revisar manualmente o CSV `golden_claims_human_review.csv` (~20 itens) para estimar o erro do juiz.
2. Rodar `python scripts/eval_golden_claims.py` e, se possível, com `--nli`. Ajustar prompts e limiares `claims_*` conforme os ICs.
3. Montar um conjunto de retrieval com **laudos reais** (pares alegação→URL) com pelo menos 100 casos. Rodar `evaluate_retrieval.py` com `HYBRID_ENABLED=true` e `false`, comparando recall@k, MRR e nDCG com IC.
4. Calibrar o limiar: `python scripts/calibrate_threshold.py --min-precision 0.9`, por provedor/modelo de embedding.
5. Testar `RERANKER_ENABLED=true` e `NER_BACKEND=spacy` (`pip install -r requirements-nlp-advanced.txt`, `python -m spacy download pt_core_news_lg`) e medir ganho versus latência.
6. Habilitar `NLI_ENABLED=true` em modo só-registro, medir a concordância LLM×NLI e só então decidir sobre abstenção por discordância.
7. Acompanhar o efeito da temperatura 0.0 no `eval_generation.py` antes do deploy.
8. Revisar o `retrieval_report.json` e os dados em `checa-ai-data`, adicionando os repositórios ao workspace.
