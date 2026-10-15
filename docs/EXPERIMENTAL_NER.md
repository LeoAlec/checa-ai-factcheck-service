# NER experimental com spaCy

Esta integração reforça a guarda que impede uma evidência sobre uma entidade
de ser usada para contradizer uma alegação sobre outra entidade. Ela não muda
o contrato da API nem ativa o modelo spaCy por padrão.

## Configuração

- O padrão `NER_BACKEND=heuristic` mantém o comportamento existente.
- Para experimentar, instale `requirements-ner.txt`, baixe o modelo
  `pt_core_news_lg` e use `NER_BACKEND=spacy`.
- Se o modelo não estiver disponível, o serviço faz fallback para a heurística.

Antes de promover o backend spaCy, medir cobertura de entidades, falsos bloqueios
e latência no fluxo completo.
