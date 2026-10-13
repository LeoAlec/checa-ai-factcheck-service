# Estado de validação dos prompts no Langfuse

O serviço mantém dois prompts independentes e auditáveis:

| Prompt | Versão local | Hash local |
| --- | --- | --- |
| `evidence-flow` | `evidence-flow-v3` | registrado em `src/rag/prompts.lock.json` |
| `claim-analysis` | `claim-analysis-v1` | registrado em `src/rag/prompts.lock.json` |

Ambos têm fallback local, são sincronizados pelo mesmo `scripts/sync_prompts.py`
e aparecem nos metadados do trace (versão e hash). A extração de alegação usa seu
próprio bundle durante a chamada à LLM; ela não herda o prompt de síntese.

## Última verificação remota

A verificação somente leitura executada nesta máquina não conseguiu consultar o
Langfuse: `LANGFUSE_ENABLED` está desabilitado e as chaves Langfuse não estavam
disponíveis ao processo. Portanto, a versão remota ativa **não foi confirmada** e
nenhum label ou prompt foi alterado.

Quando as chaves forem disponibilizadas, valide sem escrita:

```powershell
$env:LANGFUSE_ENABLED = "true"
python scripts/sync_prompts.py --check --label production
```

O resultado deve indicar, para os dois nomes, uma versão remota com texto idêntico
ao local ou que o label `production` seria movido. Só use o comando sem `--check`
ou `--promote` após autorizar a alteração no Langfuse.
