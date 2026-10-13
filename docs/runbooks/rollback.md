# Rollback em produção (VM Azure)

Acesso administrativo ao Qdrant: `ssh -L 6333:127.0.0.1:6333 <user>@<host>`.

## App (imagem)
- **Automático:** o workflow *Deploy* reverte `APP_IMAGE` se o smoke test falhar (o job fica vermelho).
- **Manual:** Actions → *Deploy* → *Run workflow* com `ref` = SHA anterior. Mais rápido, na VM:
  edite `APP_IMAGE` em `~/checa-ai/.env` para a tag antiga (continua no GHCR) e `docker compose up -d app`.

## Prompts
Republique o Release anterior ou mova o label `production` no Langfuse. Não exige redeploy.

## Corpus (DVC) e Qdrant
A ingestão é upsert: reverter `data/processed.dvc` **não** remove pontos já indexados.
1. Reverta o ponteiro: `git revert <commit>` (ou `git checkout <sha> -- data/processed.dvc`) e faça merge na `main`.
2. Actions → *Indexar corpus → Qdrant (VM)* → *Run workflow* com `recreate = true`.
   O workflow tira um snapshot antes de recriar.

### Restaurar um snapshot do Qdrant (via túnel)
```bash
C=<nome-da-colecao>
curl -s localhost:6333/collections/$C/snapshots            # lista; escolha o nome
curl -X PUT "localhost:6333/collections/$C/snapshots/recover?wait=true" \
  -H 'Content-Type: application/json' \
  -d '{"location":"file:///qdrant/snapshots/'$C'/<snapshot>"}'
```
Os snapshots vivem no volume `qdrant_storage`; apagar o volume apaga os snapshots.
