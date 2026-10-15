# Contrato da API

## `POST /api/v2/debunk`

O endpoint não versionado `POST /api/debunk` permanece com o contrato legado
para compatibilidade com a interface existente.

Entrada:

```json
{
  "message": "A alegação que deve ser verificada, com pelo menos 10 caracteres.",
  "session_id": "sessao_opcional"
}
```

Resposta:

```json
{
  "status": "matched",
  "claim": "A alegação factual identificada na mensagem.",
  "conclusion": "contradita",
  "evidence_sufficient": true,
  "evidence_level": "forte",
  "explanation": "As evidências recuperadas contradizem a alegação [1].",
  "sources": [
    {
      "title": "Título da fonte",
      "domain": "exemplo.gov.br",
      "url": "https://exemplo.gov.br/documento",
      "relevant_excerpt": "Trecho efetivamente relevante para a conclusão.",
      "origin": "local",
      "published_at": "2026-01-10"
    }
  ],
  "probabilistic_assessment": null,
  "trace_id": null
}
```

Valores fechados:

| Campo | Valores |
|---|---|
| `status` | `matched`, `deep_searched`, `abstained`, `blocked` |
| `conclusion` | `sustentada`, `contradita`, `inconclusiva` |
| `evidence_level` | `forte`, `moderada`, `limitada`, `ausente` |
| `sources[].origin` | `local`, `web` |

Invariantes:

- `claim` contém a alegação normalizada que foi consultada; é `null` quando ela não pôde ser identificada.
- `inconclusiva` implica `evidence_sufficient=false`.
- Sem fontes, a suficiência é falsa e o nível é `ausente`.
- URLs públicas são HTTP(S), sem credenciais, e devem pertencer às evidências recuperadas.
- Similaridade do Qdrant não é exposta no contrato.
- `probabilistic_assessment` permanece `null`; não há TF-IDF ou BERTimbau no fluxo de produção.

Erros de validação retornam HTTP 422. Falhas de configuração do pipeline retornam HTTP 503. Erros inesperados retornam HTTP 500 sem detalhes internos.

## `POST /api/feedback`

Recebe `trace_id`, `value` (`up` ou `down`) e `comment` opcional. Retorna HTTP 202 quando o feedback é registrado.
