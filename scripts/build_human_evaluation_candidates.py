"""Build human-evaluation draft cases from existing Qdrant evidence, read-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient
from src.rag.qdrant import patch_qdrant_local_compat


def _text(payload: dict[str, Any]) -> str:
    return str(payload.get("texto_chunk") or payload.get("texto") or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera rascunhos de casos humanos a partir de evidências existentes, sem indexar nada."
    )
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--qdrant-url")
    location.add_argument("--qdrant-path", type=Path)
    parser.add_argument(
        "--allow-local-reference-index",
        action="store_true",
        help="confirma consulta ao índice Qdrant local de referência",
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit deve ser maior que zero")
    if args.qdrant_path and not args.allow_local_reference_index:
        raise ValueError(
            "o índice local exige --allow-local-reference-index; consulte docs/REAL_QDRANT_VALIDATION.md"
        )
    patch_qdrant_local_compat()
    client = (
        QdrantClient(url=args.qdrant_url, timeout=10)
        if args.qdrant_url
        else QdrantClient(path=str(args.qdrant_path))
    )
    points, _ = client.scroll(
        collection_name=args.collection,
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )
    drafts: list[dict[str, Any]] = []
    for index, point in enumerate(points, 1):
        payload = point.payload or {}
        excerpt = _text(payload)
        url = str(payload.get("url") or "").strip()
        if not excerpt or not url:
            continue
        drafts.append(
            {
                "case_id": f"local-draft-{index:03d}",
                "input": excerpt,
                "expected": "sustentada",
                "source_hint": {
                    "title": payload.get("titulo") or "Sem título",
                    "url": url,
                    "excerpt": excerpt,
                },
                "reviewer_instruction": (
                    "Rascunho gerado da evidência. Confirme que a frase contém uma alegação factual "
                    "clara antes de usá-la; remova casos vagos, duplicados ou sem fonte primária."
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in drafts) + ("\n" if drafts else ""),
        encoding="utf-8",
    )
    print(f"{len(drafts)} rascunhos gravados em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
