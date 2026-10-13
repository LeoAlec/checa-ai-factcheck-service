# Arquitetura da fase 1

## Componentes

- `app.py`: API FastAPI e interface web.
- `src/models/contracts.py`: contrato Pydantic compartilhado pela API e pela saída da LLM.
- `src/fact_check_service.py`: orquestra guardrails, recuperação, fallback e síntese.
- `src/rag/retriever.py`: converte hits do Qdrant, deduplica e monta o contexto.
- `src/rag/web_search.py`: consulta Tavily, valida URLs, filtra injection e prioriza fontes.
- `src/rag/guardrails.py`: valida JSON e garante que toda URL pertença às evidências.
- `src/safety/`: moderação de entrada/saída e detecção de prompt injection.

## Fluxo de decisão

```text
mensagem, notícia ou link
  -> validação/moderação/injection
  -> identificação estruturada da alegação
       -> ausente/contexto essencial faltante -> inconclusiva
       -> URL isolada -> contexto inicial da busca web para identificar a alegação
  -> Qdrant local usando a alegação identificada
       -> evidência relevante -> síntese JSON -> conclusão
       -> ausente/insuficiente -> busca web estruturada -> síntese JSON
  -> grounding de URLs + moderação da saída
  -> resposta pública
```

`matched` significa que a resposta foi concluída com fontes locais. `deep_searched` significa que o fallback web foi executado. `abstained` significa que não foi possível produzir uma conclusão sustentada. `blocked` representa uma interrupção de segurança. Nenhum desses estados equivale, sozinho, a verdadeiro ou falso.

A similaridade vetorial serve apenas para selecionar candidatos. A síntese precisa confirmar que cada fonte trata da mesma alegação — inclusive entidade, ação, data, local e negação relevantes — antes de concluir. Coincidência temática não é suporte evidencial.

## Grounding e fontes não confiáveis

Contextos locais e web são tratados como dados, delimitados e não confiáveis. Evidências com risco alto de prompt injection são excluídas antes de chegar à LLM. Depois da geração, qualquer URL fora da lista recuperada é removida. A lista pública contém somente fontes usadas e validadas, com `title`, `domain`, `url`, `relevant_excerpt` e `origin`.

## Futura classificação probabilística

O contrato já contém `probabilistic_assessment: null`. Uma implementação futura deve preencher esse objeto por uma porta independente após a validação da entrada. A orquestração poderá combinar o sinal do classificador com métricas internas, mas a conclusão evidencial continuará sendo derivada das fontes. O adaptador MLflow deverá carregar o modelo por versão/alias e registrar latência, versão e drift; nenhuma mudança é necessária nos campos atuais da resposta.

O backend de referência já organiza integrações externas como adapters e possui uma porta de classificador assíncrona no worker. A integração futura deve criar naquele projeto um adapter HTTP para este serviço, mantendo aqui o contrato evidencial independente e sem acoplar este processo a PostgreSQL, Redis, Evolution API ou WhatsApp.
