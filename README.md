# Checa AI Fact-check Service

Serviço independente para verificar alegações com evidências recuperadas. O pipeline consulta primeiro o Qdrant local e, quando a cobertura local é insuficiente, usa busca externa. A conclusão evidencial é sempre uma entre `sustentada`, `contradita` e `inconclusiva`.

O status operacional (`matched`, `deep_searched`, `abstained`, `blocked`) é deliberadamente separado da conclusão. Similaridade vetorial é apenas um sinal interno de recuperação e não faz parte da resposta HTTP.

## Execução local

Requer Python 3.11+ e Docker para o Qdrant.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d qdrant
uvicorn app:app --reload --port 8000
```

Preencha ao menos `OPENAI_API_KEY`. Para habilitar o fallback web, preencha também `TAVILY_API_KEY`. A interface fica em `http://localhost:8000/debunk`, a documentação OpenAPI em `http://localhost:8000/docs` e o healthcheck em `http://localhost:8000/health`.

Para usar embeddings locais, configure `EMBEDDING_PROVIDER=local`; a coleção precisa ter sido indexada com o mesmo modelo e dimensão utilizados na consulta.

## Testes

```powershell
python -m pytest -q
```

O fluxo HTTP ponta a ponta, com Qdrant e provedores substituídos por doubles
determinísticos, pode ser executado isoladamente:

```powershell
python -m pytest tests/test_evidence_api_e2e.py -q
```

O smoke test com Qdrant e provedores externos reais é opcional. Ele requer Docker em
execução, coleção indexada e as variáveis `OPENAI_API_KEY` e, para o fallback web,
`TAVILY_API_KEY`:

```powershell
$env:RUN_E2E = "1"
python -m pytest tests/test_safety_e2e.py -q
```

Use uma coleção dedicada, indexada com o mesmo provider, modelo e dimensão usados na
consulta. O passo a passo completo, incluindo o corpus controlado e o Langfuse opcional,
está em [Smoke E2E real](docs/LIVE_E2E.md).

## Fluxo

1. Validação Pydantic da entrada.
2. Moderação e bloqueio de prompt injection.
3. Recuperação local no Qdrant e filtragem de evidências contaminadas.
4. Busca web estruturada quando a evidência local não existe ou é insuficiente.
5. Síntese da LLM em JSON estrito.
6. Validação determinística das URLs contra as evidências recuperadas.
7. Resposta com conclusão, suficiência, nível, explicação e fontes usadas.

Veja [Arquitetura](docs/ARCHITECTURE.md) e [Contrato da API](docs/API_CONTRACT.md).

## Avaliação offline de classificadores

Existe uma bancada isolada para avaliar os candidatos TF-IDF e BERTimbau antes de qualquer integração. Ela não é importada pelo endpoint `/api/debunk`, não treina modelos e não trata probabilidade de padrão textual como probabilidade de veracidade.

1. Copie `data/evaluation/external_messages_template.csv` para uma localização fora do Git e faça curadoria humana dos splits `calibration` e `test`.
2. Configure em `.env` os caminhos externos dos artefatos e do dataset.
3. Valide e execute:

```powershell
python scripts/validate_classifier_dataset.py $env:CLASSIFIER_EXTERNAL_DATASET_PATH
python scripts/evaluate_classifiers.py
```

Para gerar um conjunto-piloto reproduzível a partir dos rótulos já existentes em `data/raw/fakeWhatsApp.BR_2018.csv`, sem uso de LLM, execute antes:

```powershell
python scripts/build_fakewhatsapp_evaluation_dataset.py
python scripts/validate_classifier_dataset.py data/evaluation/fakewhatsapp_external.csv
```

O importador usa agosto–setembro de 2018 em `calibration` e outubro em `test`, remove duplicatas exatas e mensagens acima de 5.000 caracteres. O campo `topic` indica coorte temporal, não um tema semântico inferido.

O comando gera JSON, Markdown e, quando houver avaliações, uma curva de confiabilidade em `reports/classifier-evaluation/`. Para TF-IDF, instale `requirements-classifier-evaluation.txt`; para BERTimbau, instale também `requirements-classifier-transformer.txt`. Consulte [Avaliação de classificadores](docs/CLASSIFIER_EVALUATION.md) para os arquivos esperados, metodologia e limites da recomendação.

Os totais históricos atribuídos ao FakeRecogna usam universos e filtros diferentes; veja a [auditoria dos números](docs/FAKERECOGNA_AUDIT.md). Eles não devem ser usados para comparar ou aprovar modelos sem igualar corpus, splits e métricas.

## Limitações da fase 1

- Não há NER, NLI nem classificador probabilístico integrado ao serviço; a avaliação offline não altera a produção.
- A suficiência é uma decisão evidencial estruturada da síntese, protegida por invariantes determinísticos, mas ainda requer avaliação contínua.
- A busca web depende da cobertura e disponibilidade do Tavily; resultados inválidos ou com prompt injection são descartados.
- O corpus Qdrant deve ser alimentado separadamente pelos scripts de ingestão existentes.

## Evolução prevista

O campo opcional `probabilistic_assessment` já reserva o ponto de extensão do contrato para um classificador futuro. Ele deverá atuar como sinal auxiliar, sem substituir a conclusão baseada em evidências. MLflow poderá registrar versão, métricas, artefato e lineage desse classificador sem mudar os campos principais da API.
