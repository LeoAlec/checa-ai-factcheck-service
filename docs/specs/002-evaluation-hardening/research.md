# Research: decisões de endurecimento

## Seleção de limiar

**Decision**: suportar `max_f1` para experimentos/piloto e `min_precision` para contextos conservadores.

**Rationale**: as estratégias otimizam objetivos diferentes e não devem ser representadas pelo mesmo campo implícito.

**Alternatives considered**: manter apenas `min_precision` descartaria o experimento atual; manter apenas `max_f1` não oferece controle explícito de falsos positivos.

## Promoção

**Decision**: separar `reliable` de `promotion_tier`; produção aceita somente artefatos confiáveis promovidos para produção, enquanto piloto exige opt-in.

**Rationale**: uma métrica pode ser útil experimentalmente sem satisfazer critérios de produção.

## Auditoria

**Decision**: exportar casos para revisão humana, sem classificar causas automaticamente.

**Rationale**: distinguir rótulo incorreto, ausência no corpus e falha do embedding exige julgamento e investigação do índice.

## Consistência factual

**Decision**: começar com invariantes determinísticos de fontes/conclusão, deixando juízes LLM e NLI em modo experimental futuro.

**Rationale**: invariantes são reproduzíveis, baratos e não introduzem um novo modelo não validado no caminho crítico.
