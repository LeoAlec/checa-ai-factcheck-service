# Avaliação externa de TF-IDF e BERTimbau

## Escopo e separação da produção

Este módulo compara classificadores de **padrão textual associado a conteúdo falso**. A probabilidade produzida não é probabilidade de uma alegação ser falsa e jamais substitui o fluxo evidencial. Nada deste módulo é carregado por `/api/debunk` ou pela interface de produção.

Não há treino automático. Sem pesos ou dataset externo curado, o comando gera apenas inventário e recomenda `nao_integrar_nenhum_por_enquanto`.

## Artefatos esperados

Os arquivos foram definidos pelo projeto de treinamento `checa-ai-machine-learning`, mas devem permanecer fora deste Git:

```text
artifacts/
├── baseline/
│   ├── best_model.joblib
│   ├── metrics.json
│   └── test_predictions.csv
└── transformer/<configuracao>/
    ├── best_model/
    │   ├── config.json
    │   ├── tokenizer_config.json e arquivos do tokenizer
    │   ├── model.safetensors ou pytorch_model.bin
    │   └── checa_ai_config.json
    ├── metrics.json
    └── test_predictions.csv
```

`best_model.joblib` referencia a classe histórica `checa_ai.preprocessing.PortugueseTextPreprocessor`. O avaliador fornece uma implementação compatível e requer `snowballstemmer`, instalado por `requirements-classifier-evaluation.txt`. O adaptador BERTimbau usa somente arquivos locais (`local_files_only=True`) e não baixa pesos.

Variáveis:

| Variável | Conteúdo |
|---|---|
| `CLASSIFIER_TFIDF_MODEL_PATH` | caminho de `baseline/best_model.joblib` |
| `CLASSIFIER_TFIDF_METRICS_PATH` | caminho de `baseline/metrics.json` (inventário) |
| `CLASSIFIER_TFIDF_PREDICTIONS_PATH` | caminho de `baseline/test_predictions.csv` (inventário) |
| `CLASSIFIER_BERTIMBAU_MODEL_PATH` | diretório `<configuracao>/best_model` |
| `CLASSIFIER_BERTIMBAU_METRICS_PATH` | caminho de `<configuracao>/metrics.json` (inventário) |
| `CLASSIFIER_BERTIMBAU_PREDICTIONS_PATH` | caminho de `<configuracao>/test_predictions.csv` (inventário) |
| `CLASSIFIER_EXTERNAL_DATASET_PATH` | CSV ou JSONL externo curado |
| `CLASSIFIER_EVALUATION_OUTPUT_DIR` | saída ignorada pelo Git |
| `CLASSIFIER_BERTIMBAU_DEVICE` | `cpu`, `cuda` etc. |
| `CLASSIFIER_BERTIMBAU_BATCH_SIZE` | lote de inferência |
| `CLASSIFIER_DECISION_THRESHOLD` | limiar de classe, padrão `0.5` |

## Dataset externo

O template versionado contém apenas o cabeçalho. Cada linha deve ser rotulada por um processo de curadoria rastreável, nunca por LLM:

| Campo | Regra |
|---|---|
| `id` | identificador único |
| `text` | mensagem curta original |
| `label` | `verdadeiro` ou `falso` |
| `topic` | tema usado na análise de subgrupos |
| `source` | referência auditável da decisão de rótulo |
| `published_at` | data ISO `AAAA-MM-DD` |
| `split` | `calibration` ou `test` |

Os dois splits são obrigatórios, não compartilham IDs e precisam conter as duas classes. O split de calibração deve ser externo e distinto dos dados de treino originais. O teste permanece intocado até a avaliação final.

## Metodologia

- Probabilidade bruta: classe `1` (`Falso`) do `predict_proba`/score no TF-IDF e softmax do logit de índice `1` no BERTimbau.
- TF-IDF: sigmoid/Platt scaling ajustado somente em `calibration`.
- BERTimbau: temperature scaling ajustado somente em `calibration`.
- Resultado: métricas brutas e calibradas apenas em `test`; accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier, ECE, matriz de confusão, pontos da curva de confiabilidade, recorte por tema e erros individuais.
- Custo: tamanho do artefato, tempo de carga, latência total e por mensagem no mesmo hardware. Não se inventa custo monetário para inferência local.
- Reprodutibilidade: cada artefato recebe uma versão derivada de SHA-256 do conteúdo.

O gate inicial de recomendação exige pelo menos 100 mensagens no teste, F1 >= 0,70, ROC-AUC >= 0,75 e ECE <= 0,10. É uma regra conservadora de engenharia, não um critério científico definitivo; deve ser revisada com produto e risco. Se ambos passarem, BERTimbau só justifica a complexidade adicional quando ganhar pelo menos 0,03 de F1; caso contrário, prefere-se TF-IDF.

## Ambiente de avaliação verificado

Foi validado localmente em Windows com Python 3.13, `scikit-learn==1.6.1`, `snowballstemmer==3.1.1`, `torch==2.14.1` e `transformers==4.57.6`. O `scikit-learn` é fixado em 1.6.1 porque essa é a versão registrada dentro do `best_model.joblib` fornecido.

```powershell
python -m venv .venv-classifier-evaluation
.\.venv-classifier-evaluation\Scripts\python.exe -m pip install --upgrade pip
.\.venv-classifier-evaluation\Scripts\python.exe -m pip install -r requirements-classifier-transformer.txt

$env:CLASSIFIER_TFIDF_MODEL_PATH = "models\baseline\best_model.joblib"
$env:CLASSIFIER_TFIDF_METRICS_PATH = "models\baseline\metrics.json"
$env:CLASSIFIER_TFIDF_PREDICTIONS_PATH = "models\baseline\test_predictions.csv"
$env:CLASSIFIER_BERTIMBAU_MODEL_PATH = "models\best_model_transformer"
$env:CLASSIFIER_BERTIMBAU_METRICS_PATH = "models\best_model_transformer\metrics.json"
$env:CLASSIFIER_BERTIMBAU_PREDICTIONS_PATH = "models\best_model_transformer\test_predictions.csv"
$env:CLASSIFIER_EXTERNAL_DATASET_PATH = "C:\caminho\para\external_messages.csv"

.\.venv-classifier-evaluation\Scripts\python.exe scripts\validate_classifier_dataset.py $env:CLASSIFIER_EXTERNAL_DATASET_PATH
.\.venv-classifier-evaluation\Scripts\python.exe scripts\evaluate_classifiers.py
```

O ambiente e os pesos permanecem ignorados pelo Git. Se o Windows retornar `WinError 4551` ao importar `torch` ou extensões do scikit-learn, a política corporativa precisa autorizar as DLLs assinadas no diretório `Lib\site-packages` desse ambiente virtual. Não desative ou contorne a política; peça a liberação à equipe responsável e repita a verificação local.

## Candidatos a dataset externo

O dataset precisa conter somente a mensagem alegada, não a explicação da checagem, e manter a fonte que justificou cada rótulo. Opções a avaliar:

- **FakeWhatsApp.BR 2018**: já existe em `data/raw/fakeWhatsApp.BR_2018.csv`; possui mensagens curtas e os campos `misinformation=0` e `misinformation=1`. Há 21.289 linhas rotuladas localmente (9.877 e 11.412, respectivamente) e 261.312 com `-1`, que devem ser descartadas. Antes de usar, audite licença, proveniência e duplicatas por texto/URL/grupo.
- **ClaimReview de agências de checagem brasileiras**: alegações publicadas por Aos Fatos, Agência Lupa, Boatos.org e Fato ou Fake podem formar um conjunto temporal e de fontes diferente do Fake.br. Exige extração auditável e revisão humana do texto da alegação e do veredito; não use automaticamente o corpo explicativo como entrada.
- **Conjuntos de tarefas do CheckThat!/CLEF ou de Claim Matching**: use somente subconjuntos em português que tragam rótulo binário verificável e licença compatível. Muitos desses conjuntos medem verificabilidade ou recuperação de evidência, não falsidade; descarte-os quando o contrato não corresponder.

O `FakeTrueBr_corpus.csv` local não deve ser usado diretamente como teste externo: sua coluna `true` contém explicações e evidências de checagem, o que vazaria a resposta para o classificador. O Fake.br usado no treino também não pode ser reutilizado para calibrar ou testar esta fase.

### Piloto FakeWhatsApp.BR reproduzível

O script `scripts/build_fakewhatsapp_evaluation_dataset.py` usa exclusivamente os rótulos existentes: `misinformation=0` torna-se `verdadeiro`, e `1` torna-se `falso`. Ele descarta `-1`, placeholders de mídia, textos acima de 5.000 caracteres e duplicatas exatas; duplicatas com rótulos conflitantes também são removidas. Por padrão, agosto–setembro de 2018 formam `calibration`, outubro forma `test`, e a amostragem estratificada usa seed 42.

```powershell
.\.venv-classifier-evaluation\Scripts\python.exe scripts\build_fakewhatsapp_evaluation_dataset.py
.\.venv-classifier-evaluation\Scripts\python.exe scripts\validate_classifier_dataset.py data\evaluation\fakewhatsapp_external.csv
```

O CSV e o manifesto são ignorados pelo Git. O campo `topic` é derivado como `coorte_temporal:AAAA-MM`, portanto a seção “por tema” do relatório deve ser lida como recorte temporal, não como análise temática. Os grupos de WhatsApp atravessam os períodos, uma limitação explicitamente registrada no manifesto.

### Segunda avaliação: FakeRecogna controlado

O arquivo local `data/raw/FakeRecogna.xlsx` contém títulos, categorias, datas, URLs e a coluna `Classe`. O importador usa apenas `Titulo`: o campo `Noticia` é excluído porque pode conter texto processado ou explicação e causaria vazamento. `Classe=0` é convertido em `falso` e `Classe=1` em `verdadeiro`, sem gerar rótulos novos.

Por padrão, a amostra usa somente `g1.globo.com`, um domínio que contém ambas as classes no workbook. Isso reduz um viés grave observado no conjunto completo, no qual vários domínios aparecem quase só em uma classe. Calibração termina em 2020-12-31 e teste começa em 2021-01-01.

```powershell
.\.venv-classifier-evaluation\Scripts\python.exe scripts\build_fakerecogna_evaluation_dataset.py
.\.venv-classifier-evaluation\Scripts\python.exe scripts\validate_classifier_dataset.py data\evaluation\fakerecogna_external.csv
```

O manifesto registra o hash do workbook, domínios aceitos e exclusões. Ainda não se pode afirmar independência total do treino original: a proveniência do workbook e possível sobreposição com o corpus de treino precisam de auditoria. Portanto este conjunto é uma segunda evidência de avaliação, não autorização automática de integração.

### Revisão humana da amostra FakeWhatsApp

```powershell
.\.venv-classifier-evaluation\Scripts\python.exe scripts\build_fakewhatsapp_review_sample.py
```

O arquivo resultante mantém o rótulo original oculto e cria campos para decisão, URL de evidência e notas do revisor. Siga [o protocolo de revisão](LABEL_REVIEW_PROTOCOL.md) antes de usar essa auditoria para aceitar ou rejeitar os rótulos.

## Contrato futuro

```json
{
  "model_name": "tfidf",
  "model_version": "sha256:...",
  "probability_false": 0.0,
  "calibrated": false,
  "threshold_band": "baixo",
  "limitations": []
}
```

As faixas `baixo`, `medio` e `alto` são faixas operacionais configuráveis (padrões 0,35 e 0,65), não conclusões factuais. Uma integração futura deve manter esse sinal separado da conclusão evidencial.
