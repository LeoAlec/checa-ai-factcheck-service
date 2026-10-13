# Validação real com o índice Qdrant existente

O índice de referência permanece em
`C:\Users\leozi\OneDrive\Área de Trabalho\projetos\checa-ai-ml\data\qdrant_db`.
Esta configuração não copia, reindexa, recria ou executa `upsert` nessa base.

## Compatibilidade confirmada

O `meta.json` do índice declara a coleção `fact_checks_pt_v2_openai`, distância
`Cosine` e vetor de 1.536 dimensões. Ela é compatível com o serviço quando este
usa exatamente:

| Item | Valor |
| --- | --- |
| Coleção | `fact_checks_pt_v2_openai` |
| Provedor | `openai` |
| Modelo de embedding | `text-embedding-3-small` |
| Dimensão | `1536` |
| Distância | `Cosine` |

O serviço já adota esses valores como padrão para o provedor OpenAI. A dimensão
e a presença de pontos também são verificadas pelo smoke antes de chamar a API.
O Qdrant não registra de forma nativa o nome do modelo no `meta.json`; a garantia
do modelo vem da configuração de indexação/payload e do `.env` abaixo.

## Opção A — índice local existente, sem Docker

Use somente para consulta e apenas quando nenhum outro processo estiver usando o
mesmo diretório. No `.env`, preencha:

```dotenv
OPENAI_API_KEY=<chave OpenAI>
TAVILY_API_KEY=<chave Tavily>
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_URL=
QDRANT_PATH=C:/Users/leozi/OneDrive/Área de Trabalho/projetos/checa-ai-ml/data/qdrant_db
QDRANT_COLLECTION=fact_checks_pt_v2_openai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0
LANGFUSE_ENABLED=false
LANGFUSE_CAPTURE_CONTENT=false
RUN_E2E=1
E2E_ALLOW_REFERENCE_LOCAL_INDEX=1
```

Execute `python -m pytest tests/test_safety_e2e.py -q -rs`.

O cliente local do Qdrant não executa indexação nessa rota, mas a biblioteca pode
administrar um arquivo de lock/cache do próprio mecanismo. Portanto esta opção é
sem alteração de conteúdo, não uma garantia de montagem imutável em nível de
sistema de arquivos. Não execute `scripts/ingest_to_qdrant.py`, `--recreate` ou
scripts de curadoria apontando para esse caminho.

## Opção B — Qdrant em Docker (recomendada)

Suba um servidor isolado, com volume próprio, sem montar o diretório de referência:

```powershell
docker compose -f docker-compose.qdrant.yml up -d
Invoke-RestMethod http://localhost:6333/healthz
```

No `.env`, use:

```dotenv
OPENAI_API_KEY=<chave OpenAI>
TAVILY_API_KEY=<chave Tavily>
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=fact_checks_pt_v2_openai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0
LANGFUSE_ENABLED=false
LANGFUSE_CAPTURE_CONTENT=false
RUN_E2E=1
```

Um container novo começa sem a coleção. Não monte o banco local embarcado dentro
dele: isso não é um mecanismo de importação suportado e pode alterar arquivos de
referência. Use esta opção quando a coleção já estiver disponível no servidor
Docker ou após uma importação/snapshot explicitamente aprovada. O E2E falhará de
forma segura se a coleção estiver vazia, com dimensão diferente ou sem pontos
marcados com `text-embedding-3-small`.

## Langfuse e prompts auditáveis

Langfuse é opcional. Para habilitá-lo, além dos valores acima preencha:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=<chave pública Langfuse>
LANGFUSE_SECRET_KEY=<chave secreta Langfuse>
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
PROMPT_LABEL=staging
LANGFUSE_CAPTURE_CONTENT=false
```

Há dois prompts independentes: `evidence-flow` (`evidence-flow-v3`) e
`claim-analysis` (`claim-analysis-v1`). Ambos têm hash e versão no arquivo
`src/rag/prompts.lock.json`, fallback local e versão remota no trace. Para
validar/publicar os dois juntos:

```powershell
python scripts/sync_prompts.py --write-lock
python scripts/sync_prompts.py --check --label staging
python scripts/sync_prompts.py --label staging
python scripts/sync_prompts.py --promote
```

Os dois últimos comandos escrevem no Langfuse; execute-os somente com autorização
para publicar/mover labels. Nunca coloque chaves no repositório ou na linha de
comando.
