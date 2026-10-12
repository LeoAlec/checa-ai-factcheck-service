# Smoke E2E com serviços reais

O smoke real usa OpenAI para moderação, embeddings, extração da alegação e síntese;
Qdrant via HTTP para recuperação local; e Tavily para fallback web. Ele nunca deve
ser executado com credenciais em linha de comando ou versionadas no Git.

## Configuração

1. Crie o ambiente e instale as dependências:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

2. No arquivo local `.env`, configure sem compartilhar os valores:

   ```dotenv
   OPENAI_API_KEY=<chave com acesso a chat, moderação e embeddings>
   TAVILY_API_KEY=<chave válida do Tavily>
   EMBEDDING_PROVIDER=openai
   EMBEDDING_MODEL=text-embedding-3-small
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=checa_ai_live_e2e_openai_1536
   LLM_MODEL=gpt-4o-mini
   LLM_TEMPERATURE=0
   LANGFUSE_ENABLED=false
   LANGFUSE_CAPTURE_CONTENT=false
   ```

   A coleção deve ter vetores `COSINE` de 1.536 dimensões, pois foi indexada com
   `text-embedding-3-small`. Consulta e indexação precisam usar exatamente o mesmo
   `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` e `QDRANT_COLLECTION`.

3. Inicie e valide o Qdrant:

   ```powershell
   docker compose up -d qdrant
   docker compose ps
   Invoke-RestMethod http://localhost:6333/healthz
   ```

4. Crie a coleção E2E dedicada e indexe o corpus mínimo. `--recreate` apaga somente
   a coleção indicada em `QDRANT_COLLECTION`; não use o nome de uma coleção real:

   ```powershell
   python scripts/ingest_to_qdrant.py `
     --input tests/fixtures/live_e2e_corpus.jsonl `
     --require-docker `
     --recreate
   ```

   O script lê `QDRANT_COLLECTION`, `QDRANT_URL`, `EMBEDDING_PROVIDER` e
   `EMBEDDING_MODEL` diretamente do `.env`.

5. Execute o smoke:

   ```powershell
   $env:RUN_E2E = "1"
   python -m pytest tests/test_safety_e2e.py -q -rs
   ```

Os casos cobrem evidência local favorável, fallback web contraditório, mensagem sem
contexto, URL isolada e indisponibilidade do provedor externo. As mensagens podem ser
sobrescritas por `E2E_LOCAL_SUPPORTED_CLAIM`, `E2E_WEB_CONTRADICTED_CLAIM`,
`E2E_STANDALONE_URL` e `E2E_EXTERNAL_FAILURE_CLAIM`.

## Langfuse opcional

O Langfuse não é requisito para o resultado funcional. Para validá-lo no mesmo smoke,
configure `LANGFUSE_ENABLED=true`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
`LANGFUSE_BASE_URL`, `LANGFUSE_TRACING_ENVIRONMENT` e `PROMPT_LABEL`. Mantenha
`LANGFUSE_CAPTURE_CONTENT=false` para não enviar mensagens e evidências em claro.
Sem essas chaves, deixe `LANGFUSE_ENABLED=false`; o pipeline opera com o prompt local.
