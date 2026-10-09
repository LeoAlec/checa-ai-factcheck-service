# Integração da API de verificação com a main e o DeepSearch

**Data:** 2026-10-08
**Estado:** desenho aprovado em conversa; documento aguardando revisão do usuário

## Objetivo

Atualizar a branch `design/ml-backend-integration` com a `main` atual sem perder a
API de verificação para o backend nem o fluxo de verificação, segurança e
observabilidade incorporado à `main`. A rota `POST /api/v1/verify` ainda não foi
publicada na `main`; seu contrato v1 pode ser ampliado neste PR para entregar
também a pesquisa web cautelosa do DeepSearch/Tavily. A rota `POST /api/debunk`
continua servindo a interface web com o comportamento da `main`.

Esta entrega altera apenas `checa-ai-ml`. O adaptador do repositório irmão
`checa-ai` deverá consumir o contrato v1 ampliado em sua própria implementação.
Não fazer push como parte da resolução local sem revisar o resultado com o usuário.

## Contexto

A branch partiu de `dea30a1` e acrescentou `/api/v1/verify`, readiness, um
container FastAPI e geração curta fundamentada em até duas fontes para respostas
de WhatsApp de até 1.000 caracteres. Desde então, `origin/main` chegou a
`c58a22d` com moderação de entrada, detecção de prompt injection, exclusão de
chunks de alto risco, validação de saída, grounding de URLs, abstinência indicada
pelo modelo, fallback de busca web, tracing Langfuse, gestão de prompts e CI.
O merge tem quatro conflitos textuais: `.env.example`, `README.md`, `app.py` e
`src/fact_check_service.py`. O último concentra o conflito funcional: a branch
adicionou `verify_claim_for_backend`, enquanto a `main` reestruturou a busca e a
geração. A imagem da API instala `requirements-api.txt`, lista separada das
dependências da `main`; ela ainda não inclui Tavily e Langfuse.

Na `main`, o DeepSearch já devolve o status `deep_searched` para `/api/debunk`,
mas devolve `sources: []` e um texto sem citações estruturadas. Fontes web
citadas na v1, o campo `source_type` e seu orçamento de resposta são adaptações
novas exigidas pelo contrato ampliado aprovado, não comportamento já existente
na `main`.

## Fluxo de verificação

1. Aplicar os guardrails de entrada da `main` antes da busca.
2. Recuperar evidências do Qdrant usando os filtros e a ordenação da `main`,
   inclusive a exclusão de chunks com risco alto de prompt injection.
3. Quando houver laudos suficientes, gerar resposta curta fundamentada para v1.
   Manter a validação de citações numeradas, limite de fontes, URLs seguras e
   orçamento de caracteres da branch. Aplicar também validação de segurança de
   saída e observabilidade da `main`.
4. Quando o score for insuficiente ou o modelo indicar que os laudos não tratam
   da alegação, usar o DeepSearch/Tavily da `main`. Resumir seus resultados de
   forma neutra, sem veredito factual ou crédito de checagem oficial. A v1 deve
   devolver apenas URLs HTTP(S) presentes nos resultados da busca e citadas por
   número no resumo. Tratar títulos e snippets web como dados não confiáveis.
   Limitar o tamanho da resposta, sem cortar URLs ou ressalvas.
5. Sem evidência adequada no corpus nem resultado web útil, devolver evidência
   insuficiente. Uma entrada bloqueada gera recusa explícita, sem iniciar busca.

Os componentes existentes de prompt, tracing e cliente OpenAI da `main` devem
continuar sendo usados onde se aplicarem. A geração curta da v1 usa saída
estruturada para reconhecer `laudos_tratam_alegacao: false` e instruções
próprias para o contrato do backend; ela não deve ser vinculada indevidamente à
versão do prompt `debunk` do fluxo web. A integração deve compartilhar as etapas
de busca e segurança e manter a apresentação de cada rota separada. A adaptação
da v1 para fontes web não deve mudar o corpo de resposta de `/api/debunk`.

## Contrato HTTP v1 ampliado

`POST /api/v1/verify` recebe `{ "text": "..." }`, com 1 a 10.000 caracteres e
ao menos um caractere não branco. A resposta conserva `schema_version: 1`,
`verdict`, `similarity_score`, `counter_narrative` e `sources`.

| `verdict` | Significado | Narrativa | Fontes | Score |
|---|---|---|---|---|
| `matched` | Laudos do corpus sustentam a resposta | Curta e com citações verificadas | 1–2 laudos HTTP(S) citados | Melhor similaridade do corpus |
| `deep_searched` | Busca web gerou contexto cauteloso, sem checagem oficial | Resumo neutro com ressalva | URLs HTTP(S) da Tavily usadas no resumo | Melhor similaridade do corpus, possivelmente 0 |
| `insufficient_evidence` | Nem corpus nem web sustentam resposta útil | `null` | `[]` | Melhor similaridade do corpus, possivelmente 0 |
| `blocked` | Guardrail impediu a solicitação ou a saída | Recusa segura | `[]` | `null` se a entrada foi bloqueada antes da busca |

Cada fonte mantém `title`, `domain` e `url`; `source_type` distingue
`fact_check` de `web`. O veredito também distingue a procedência da resposta.
`similarity_score` é um dado de auditoria do Qdrant, nunca probabilidade de
verdade nem avaliação de confiabilidade da web. Uma resposta `deep_searched`
não pode ser rotulada ou formatada como `matched` pelo consumidor.

A contranarrativa e os links selecionados devem caber no orçamento reservado
pelo contrato atual para uma mensagem final do backend de 1.000 caracteres.
Nenhuma URL ou aviso é truncado silenciosamente. Uma resposta `matched` ou
`deep_searched` requer ao menos uma fonte usada e visível. O resumo web deve
preservar o aviso de cautela do fluxo da `main`.

## Falhas e compatibilidade

- Entrada inválida retorna 422. Dependências indisponíveis retornam 503;
  falhas da geração fundamentada para `matched` ou violação do contrato
  retornam 5xx, sem veredito falso.
- Ausência de resultados Tavily ou erro na síntese web segue a abstenção da
  `main`. Erro de busca web não é convertido em checagem encontrada.
- Resposta bloqueada não é mapeada para `insufficient_evidence`, porque bloqueio
  não informa nada sobre a disponibilidade de laudos.
- `/api/debunk`, `/api/feedback`, `/health` e o shutdown de tracing da `main`
  preservam seus contratos. `/ready` mantém a verificação de chave OpenAI,
  Qdrant, coleção e dimensão do embedding.
- O backend que espera somente dois vereditos precisará atualizar sua validação,
  checkpoint e formatação antes de consumir a v1 ampliada. Não adaptar esse
  repositório irmão nesta entrega.

## Integração Git e implantação

Fazer merge de `origin/main` na branch local e resolver cada conflito pela
intenção das duas alterações. Preservar workflows, prompts e dependências da
`main`, além de Dockerfile, Compose e dependências específicas da API da branch.
Consolidar README e `.env.example`, incluindo configuração de Tavily e Langfuse
necessária ao fluxo integrado. Completar `requirements-api.txt` com as
dependências importadas no caminho de startup e verificação da API e propagar
as configurações necessárias pelo Compose. Não registrar alegações, snippets,
laudos ou chaves em logs do caminho integrado. Não reescrever o histórico nem
publicar a branch automaticamente.

## Critérios de aceite

1. O merge local não deixa marcadores de conflito, arquivos não resolvidos ou
   perda das rotas e componentes de qualquer branch.
2. Testes verificam os quatro vereditos v1, bloqueio antes da busca, filtragem
   de chunks perigosos e fallback web por score baixo ou abstinência do modelo.
3. `matched` retorna apenas laudos citados; `deep_searched` retorna apenas
   resultados Tavily usados, com procedência e ressalva claras. URLs inventadas,
   não HTTP(S) e respostas sem fonte válida são rejeitadas ou não retornadas como
   resposta fundamentada.
4. Os limites de tamanho, a concorrência da rota, readiness e os erros sem
   veredito são cobertos por testes relevantes.
5. Os testes existentes de `/api/debunk`, guardrails, prompts, observabilidade,
   API v1 e pipeline RAG passam na árvore integrada. A configuração Compose é
   validada; a imagem inicia com as dependências do fluxo da `main` e as
   configurações de Tavily e Langfuse são repassadas quando presentes. Build e
   teste operacional dependentes de Docker/Qdrant são executados quando o
   ambiente estiver disponível; qualquer limitação é relatada com precisão.
