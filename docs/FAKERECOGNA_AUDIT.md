# Auditoria dos números atribuídos ao FakeRecogna

Os números encontrados nos artefatos das fases anteriores não descrevem o mesmo
conjunto de avaliação. Por isso, não são diretamente comparáveis e não autorizam a
escolha de um modelo.

| Artefato | Universo | Entrada | Registros |
|---|---|---|---:|
| `notebooks/eda_comparativa.ipynb` | FakeRecogna Excel | `Noticia`, com fallback para `Titulo` | 11.902 (5.951 por classe) |
| `notebooks/eda_comparativa.ipynb` | Consolidado de FakeRecogna + FakeTrueBr + FakeWhatsApp | mistura dos três corpora | 36.773; 22.410 textos únicos |
| `notebooks/eda_comparativa_extrativo.ipynb` | FakeRecogna extrativo | corpus extrativo distinto | 52.800 brutos; 52.741 válidos; 52.603 únicos |
| `data/evaluation/fakerecogna_external.manifest.json` | avaliação controlada atual | somente `Titulo`, domínio G1 e corte temporal | 11.904 linhas lidas; 11.902 rotuladas; 750 calibração + 400 teste exportadas |
| `checa-ai-machine-learning/docs/relatorio_tecnico.md` | Fake.br-Corpus | texto completo | 7.200 documentos; 5.080/1.110/1.010 em treino/validação/teste |

O workbook `FakeRecogna.xlsx` presente neste serviço e no repositório de referência
`checa-ai-ml` tem o mesmo SHA-256:
`00e70d8173fc6b06fe0a387bb7aa8078d2e64d69df86b8971817baa901ed874a`.
Assim, a divergência de 11.902 versus 11.904 é de regra de contagem: o primeiro valor
considera somente linhas rotuladas; o segundo conta todas as linhas de dados lidas.

As diferenças maiores vêm de mudança de universo e protocolo: o total 36.773 é
consolidado, 52.800 pertence ao corpus extrativo, 7.200 pertence ao Fake.br-Corpus e
1.150 é uma amostra de avaliação filtrada. Também mudam o campo de entrada, a
deduplicação, o filtro de domínio e os splits (por grupos, por data ou calibração/teste).
Métricas só podem ser comparadas depois de confirmar o mesmo hash/manifesto, campo de
texto, mapeamento de rótulos, deduplicação, splits e definição da métrica.
