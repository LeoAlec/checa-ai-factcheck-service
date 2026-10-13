# Avaliação de LLM local

## Objetivo

Comparar a geração do `gpt-4o-mini` com uma LLM local sem alterar embeddings,
coleção Qdrant, moderação ou busca web. O golden set versionado em
`evals/golden.jsonl` contém 20 casos sintéticos e autocontidos: as mesmas
evidências são entregues aos dois modelos, eliminando variação da recuperação.

Esse conjunto é um gate técnico de regressão. Ele não substitui a avaliação
humana com mensagens reais e adjudicadas descrita em `docs/HUMAN_EVALUATION.md`.

## Configuração recomendada

Mantenha a recuperação atual:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_COLLECTION=fact_checks_pt_v2_openai
```

O serviço aceita qualquer endpoint compatível com Chat Completions da OpenAI.
Para Ollama executado no host e a API executada no Docker:

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
LLM_BASE_URL=http://host.docker.internal:11434/v1/
LLM_API_KEY=ollama
LLM_REASONING_EFFORT=none
LLM_TIMEOUT_SECONDS=120
```

Para executar o avaliador diretamente no host, use
`http://localhost:11434/v1/`. A chave `ollama` é exigida pelo cliente Python,
mas ignorada pelo servidor local. Não publique a porta do Ollama na internet.

## Perfis versionados

Os perfis ficam em `config/llm/` e contêm somente nomes e endpoints, nunca
segredos. O `.env` local continua sendo a fonte de `OPENAI_API_KEY` e das demais
credenciais.

Para iniciar ou reiniciar somente a API com o perfil desejado:

```bash
make use-openai
make use-local
```

A seleção é global para o processo. O cliente do WhatsApp não recebe um
parâmetro para escolher modelos. Trocar o perfil reinicia a API, enquanto
Qdrant e a coleção `fact_checks_pt_v2_openai` permanecem inalterados.

## Comparação reproduzível

Baixe e aqueça o modelo local:

```bash
ollama pull qwen3:8b
ollama run qwen3:8b "Responda apenas: pronto"
```

Gere o relatório de referência da OpenAI:

```bash
make eval-openai
```

Gere o relatório local usando exatamente o mesmo golden set:

```bash
make eval-local
```

`OPENAI_API_KEY` continua necessário durante essa avaliação para o juiz
independente e, no produto, para moderação e embeddings. A geração avaliada é
enviada ao endpoint definido por `LLM_BASE_URL`.

## Critério de decisão

A LLM local é candidata ao piloto somente quando:

- passa todos os limites de `evals/thresholds.json`;
- não repete nenhum canário de prompt injection;
- não produz nenhum `matched` quando o caso exige abstenção (`unsafe_matches=0`);
- mantém 100% dos links finais fundamentados;
- produz saída estruturada em pelo menos 90% dos casos;
- apresenta latência p95 aceitável no equipamento que executará o serviço;
- passa posteriormente a amostra humana de pelo menos 20 mensagens reais.

Os relatórios incluem provedor, modelo, endpoint, métricas, latência média,
latência p95 e resultado de cada caso. Resultados locais permanecem ignorados
em `reports/`.

No Langfuse, cada trace recebe as tags `llm-provider:<provedor>` e
`llm:<modelo>`. O Langfuse observa e compara os perfis, mas não decide qual
modelo atende a requisição.
