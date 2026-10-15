# Feature Specification: Endurecimento da avaliação de retrieval

**Feature Branch**: `feat/nlp-review-retrieval-eval`

**Created**: 2026-10-14

**Status**: Approved for implementation

**Input**: Consolidar as recomendações da revisão de PLN: calibração reproduzível, promoção segura de limiares, auditoria dos falsos positivos e guardrails rastreáveis.

## User Scenarios & Testing

### User Story 1 - Promover limiares com segurança (Priority: P1)

Como responsável pelo serviço, quero distinguir limiares experimentais de limiares aprovados para que um resultado não confiável não altere silenciosamente o comportamento de produção.

**Why this priority**: O limiar controla diretamente associações incorretas entre alegações e laudos.

**Independent Test**: Um artefato não confiável ou de tier incompatível é ignorado, enquanto um artefato aprovado para o tier configurado é carregado.

**Acceptance Scenarios**:

1. **Given** um limiar marcado como piloto, **When** o serviço inicia no tier de produção, **Then** o valor não é aplicado.
2. **Given** um benchmark com splits de calibração e teste, **When** o limiar é selecionado, **Then** somente a calibração influencia a escolha.
3. **Given** uma estratégia `max_f1` ou `min_precision`, **When** a calibração é executada, **Then** a estratégia e sua meta ficam registradas no artefato.

---

### User Story 2 - Auditar erros de retrieval (Priority: P2)

Como cientista de dados, quero exportar falsos positivos para revisão humana com uma taxonomia consistente, de modo que seja possível distinguir erro de rótulo, corpus e modelo.

**Why this priority**: Os dez falsos positivos conhecidos são a evidência mais útil para orientar melhorias.

**Independent Test**: Um relatório de avaliação produz uma planilha de revisão contendo apenas falsos positivos e campos suficientes para decisão humana.

**Acceptance Scenarios**:

1. **Given** um relatório com positivos e negativos, **When** a auditoria é gerada, **Then** somente negativos respondidos acima do limiar são exportados.
2. **Given** uma revisão humana preenchida, **When** o arquivo é validado, **Then** categorias desconhecidas ou decisões incompletas são rejeitadas.

---

### User Story 3 - Rastrear bloqueios e consistência (Priority: P2)

Como operador, quero que todo bloqueio de entrada ou saída seja registrado e que inconsistências factuais básicas sejam detectadas antes da resposta.

**Why this priority**: Moderação de toxicidade não substitui verificação de consistência com as evidências.

**Independent Test**: Os fluxos de saída bloqueada emitem log estruturado e o validador rejeita conclusões factuais sem evidências utilizadas.

**Acceptance Scenarios**:

1. **Given** uma saída reprovada pela moderação, **When** o serviço bloqueia a resposta, **Then** estágio e motivo são registrados sem expor o conteúdo.
2. **Given** uma conclusão sustentada ou contradita sem fontes utilizadas, **When** a resposta é validada, **Then** ela é convertida em inconclusiva.

### Edge Cases

- Artefato ausente, JSON inválido, modelo diferente ou hash do dataset ausente.
- Benchmark vazio, split desconhecido ou sem exemplos positivos.
- Relatório sem falsos positivos.
- Ambiente de piloto explicitamente configurado e ambiente de produção sem configuração.

## Requirements

### Functional Requirements

- **FR-001**: O sistema MUST oferecer seleção de limiar por `max_f1` e `min_precision`.
- **FR-002**: A seleção MUST usar exclusivamente casos do split de calibração.
- **FR-003**: O artefato MUST registrar dataset, hash, tamanhos dos splits, estratégia, métricas, coleção, embedding, commit e tier de promoção.
- **FR-004**: O runtime MUST ignorar artefatos não promovidos para o tier configurado.
- **FR-005**: O sistema MUST exportar falsos positivos para auditoria humana sem inferir automaticamente a causa.
- **FR-006**: A auditoria MUST aceitar apenas categorias documentadas e exigir decisão humana explícita.
- **FR-007**: Todo bloqueio de saída MUST produzir evento de log estruturado sem conteúdo sensível.
- **FR-008**: Conclusões factuais MUST possuir pelo menos uma evidência utilizada e validada.
- **FR-009**: Todas as mudanças MUST possuir testes automatizados sem dependência de rede.

### Key Entities

- **CalibrationArtifact**: Resultado versionado da seleção e validação de um limiar.
- **RetrievalAuditCase**: Falso positivo preparado para revisão humana.
- **PromotionTier**: Contexto em que um artefato pode ser usado (`pilot` ou `production`).

## Success Criteria

### Measurable Outcomes

- **SC-001**: Nenhum artefato `reliable=false` altera o limiar de produção sem opt-in explícito.
- **SC-002**: 100% dos artefatos novos permitem verificar o hash e o tamanho dos splits usados.
- **SC-003**: 100% dos falsos positivos do relatório podem ser exportados e classificados por humanos.
- **SC-004**: 100% dos caminhos conhecidos de bloqueio de saída emitem registro auditável.
- **SC-005**: A suíte unitária das funcionalidades novas passa sem serviços externos.

## Assumptions

- `0.65` permanece elegível para piloto, mas não é promovido automaticamente para produção.
- O benchmark atual é preservado como v1; ampliação e nova anotação humana são trabalho posterior.
- NLI, NER, reranker e alterações do contrato HTTP permanecem fora deste incremento.
