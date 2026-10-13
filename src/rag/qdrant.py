"""
src/rag/qdrant.py
─────────────────────────────────────────────────────────────────────
Cliente Qdrant compartilhado entre API e scripts.

Por padrão tenta conectar ao Qdrant server em QDRANT_URL (Docker) e,
quando permitido, cai para o modo local em QDRANT_PATH.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import QDRANT_PATH, QDRANT_URL

log = logging.getLogger(__name__)


def patch_qdrant_local_compat() -> None:
    """Suporta índices locais gerados por versões do Qdrant com campos extras no meta.json."""
    try:
        import qdrant_client.local.qdrant_local as ql

        if getattr(ql, "_compat_patched", False):
            return
        orig_create = ql.rest_models.CreateCollection

        def _safe_create_collection(**kw):
            kw.pop("payload", None)
            kw.pop("metadata", None)
            if isinstance(kw.get("vectors"), dict):
                kw["vectors"].pop("memory", None)
            return orig_create(**kw)

        ql.rest_models.CreateCollection = _safe_create_collection
        ql._compat_patched = True
    except Exception:
        pass


def get_qdrant_client(
    *,
    url: str | None = QDRANT_URL,
    path: Path = QDRANT_PATH,
    api_key: str | None = None,
    local_fallback: bool = True,
    timeout: int = 5,
) -> QdrantClient:
    """Cria cliente Qdrant preferindo servidor Docker/HTTP."""
    qdrant_url = (url or "").strip()
    if qdrant_url:
        try:
            client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=timeout)
            client.get_collections()
            log.info("Conectado ao Qdrant server em %s", qdrant_url)
            return client
        except Exception as exc:
            if not local_fallback:
                raise RuntimeError(
                    f"Qdrant server indisponível em {qdrant_url}. "
                    "Suba o container com `docker compose up -d qdrant`."
                ) from exc
            log.info("Qdrant server indisponível em %s; usando modo LOCAL em '%s'", qdrant_url, path)

    patch_qdrant_local_compat()
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))
