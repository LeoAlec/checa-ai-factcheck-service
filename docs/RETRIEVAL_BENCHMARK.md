# Benchmark de retrieval e decisões (híbrido e limiar)

Índice avaliado: `fact_checks_pt_v2_openai` (`text-embedding-3-small`, ~5,8 mil laudos Silver).

## Conjunto

`data/eval/retrieval_benchmark_120_paraphrased.jsonl`, 120 casos:

| Tipo | n | Origem |
|---|---:|---|
| matched | 60 | 19 do `retrieval_eval.jsonl` + 41 títulos FakeRecogna com laudo no índice |
| abstained, fora de domínio | 12 | consultas benignas (receita, clima, código) |
| abstained, notícia real sem laudo | 28 | FakeRecogna `Classe=1`, URL fora do índice |
| abstained, boato sem laudo no índice | 20 | FakeTrueBr, URL fora do índice |

As consultas `matched` foram **parafraseadas no estilo WhatsApp** (`scripts/paraphrase_benchmark.py`,
gpt-4o-mini). Sem isso a consulta era o próprio título do laudo indexado, o que inflava o recall
e favorecia o BM25. Cada caso tem `split` (calibration/test, 50/50 estratificado por status).

Gerar: `scripts/build_retrieval_benchmark.py` → `scripts/paraphrase_benchmark.py`.
Avaliar: `evaluate_retrieval.py` duas vezes (`HYBRID_ENABLED=false|true`) →
`scripts/analyze_retrieval_ab.py`.

## Resultados

### Híbrido (BM25 + RRF) vs. denso — pareado, 60 matched

| | Denso | Híbrido |
|---|---:|---:|
| MRR | 0,933 | 0,922 |
| recall@5 | 0,967 | 0,967 |

Diferença de MRR (híbrido − denso): −0,011, IC95% pareado [−0,064; +0,042].
4 melhoras e 5 pioras; teste de sinal p = 1,0. **Sem ganho → desligado por padrão.**

Na versão não parafraseada o híbrido parecia ajudar (5 x 2): era efeito do vazamento lexical.

### Limiar (denso), escolhido no split calibration e medido no split test

| Limiar | Precisão | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|
| 0,60 (antigo) | 0,634 | 0,867 | 0,732 | 15 | 4 |
| **0,65 (calibrado)** | **0,722** | **0,867** | **0,788** | **10** | **4** |

Acerto = laudo esperado em 1º lugar e score ≥ limiar. FP inclui responder com laudo errado.
O valor está em `evals/calibrated_thresholds.json` e só vale se `SIMILARITY_THRESHOLD` não
estiver definido no `.env`. O artefato está promovido somente para `pilot`, pois não
atinge o gate conservador de precisão 0,90. Para reproduzir esse comportamento é
necessário configurar `RETRIEVAL_CALIBRATION_TIER=pilot`; o tier padrão de produção
ignora artefatos não confiáveis.

O artefato registra 60 casos de calibração e 60 de teste, o SHA-256 do dataset,
coleção, embedding, estratégia e commit. O limiar 0,65 foi escolhido por maior F1
exclusivamente nos 60 casos de calibração. As métricas acima correspondem aos 60
casos cegos de teste.

### Auditoria dos falsos positivos

`scripts/audit_retrieval_errors.py` exporta os dez falsos positivos do split de
teste para `reports/retrieval/false_positive_review.csv`. A planilha deixa causa,
decisão e justificativa vazias para revisão humana. Categorias aceitas incluem erro
de rótulo, lacuna do corpus, divergência de entidade/tempo, coincidência apenas
temática, URL duplicada/canônica, consulta ambígua e erro do embedding.

## Limitações

- n ≈ 60 por split: os ICs são largos; 0,65 é uma melhora, não um ótimo.
- Paráfrases geradas por LLM, sem revisão humana completa. Alguns casos de origem são fracos
  (texto lematizado do FakeRecogna, títulos de resumo como "Veja o que é #FATO ou #FAKE…").
- Os negativos não foram auditados: uma "notícia real" pode ter laudo no índice sob outra URL.
  Mesmo em 0,65 restam 10 FP no teste; revisar esses casos é o próximo passo.
- Só retrieval: o efeito no veredito final (LLM) não foi medido.
- O tier `pilot` é um opt-in operacional e não equivale a promoção para produção.
